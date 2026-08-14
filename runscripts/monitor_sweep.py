"""Live, human-readable progress overview for a running (or finished) HPO sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from autonnunet.utils.paths import AUTONNUNET_OUTPUT

STALE_AFTER = timedelta(minutes=120)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "autonnunet_monitor"


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Read a CSV that may be mid-write by the sweeper (plain overwrite, not atomic);
    treat any parse failure as "try again next refresh" rather than an error.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    return df if len(df) > 0 else None


def _latest_mtime(job_dir: Path, filenames: tuple[str, ...]) -> datetime | None:
    """Newest mtime among the given files in a job dir, or None if none exist.

    progress.csv is rewritten every completed epoch; train.log is appended to
    at whatever cadence the trainer logs at, which can lag well behind actual
    progress (e.g. long gaps during validation, or just less frequent print
    statements) - using train.log alone can make an actively-running job look
    falsely stale. Taking the max of both gives the true "last real activity".
    """
    mtimes = [(job_dir / name).stat().st_mtime for name in filenames if (job_dir / name).exists()]
    return datetime.fromtimestamp(max(mtimes)) if mtimes else None


def _find_most_recent_activity(sweep_dir: Path) -> datetime | None:
    """Newest mtime across all per-job progress.csv/train.log files.

    These are rewritten during active training, unlike runhistory.csv/
    incumbent_*.csv which only update once a full iteration (all folds of one
    config) finishes - which can legitimately take hours.
    """
    latest = None
    for job_dir in sweep_dir.glob("[0-9]*"):
        if not job_dir.is_dir():
            continue
        mtime = _latest_mtime(job_dir, ("progress.csv", "train.log"))
        if mtime is not None and (latest is None or mtime > latest):
            latest = mtime
    return latest


@dataclass
class _EpochProgress:
    n_epochs: int | None
    avg_sec: float | None  # mean seconds/epoch over the whole run so far
    dice: float | None = None  # ema_fg_dice - smoothed pseudo-Dice, same metric as the
                                # baseline-comparison panel's "same epoch" figures
    train_loss: float | None = None
    val_loss: float | None = None


def _current_epoch(job_dir: Path) -> _EpochProgress:
    """Current training progress for a job, from progress.csv if it exists yet
    (rewritten once per completed epoch), else by scraping the last "Epoch N"
    line from train.log (covers epoch 0, before progress.csv appears - only
    n_epochs is available at that point, everything else stays None).
    """
    progress_path = job_dir / "progress.csv"
    if progress_path.exists():
        df = _read_csv_safe(progress_path)
        if df is not None and "Epoch" in df.columns:
            n_epochs = int(df["Epoch"].iloc[-1]) + 1  # 0-indexed in the file
            durations = df["epoch_end_timestamps"] - df["epoch_start_timestamps"]
            avg_sec = float(durations.mean()) if len(durations) > 0 else None
            last = df.iloc[-1]
            return _EpochProgress(
                n_epochs=n_epochs,
                avg_sec=avg_sec,
                dice=float(last["ema_fg_dice"]) if "ema_fg_dice" in df.columns else None,
                train_loss=float(last["train_losses"]) if "train_losses" in df.columns else None,
                val_loss=float(last["val_losses"]) if "val_losses" in df.columns else None,
            )

    train_log = job_dir / "train.log"
    if train_log.exists():
        # Log lines look like "[2026-07-30 10:48:07][Trainer][INFO] - Epoch 0"
        matches = re.findall(r"Epoch (\d+)\s*$", train_log.read_text(errors="ignore"), re.MULTILINE)
        if matches:
            return _EpochProgress(n_epochs=int(matches[-1]), avg_sec=None)
    return _EpochProgress(n_epochs=None, avg_sec=None)


def _find_in_progress_jobs(sweep_dir: Path, max_jobs: int = 10) -> list[Path]:
    """Job directories that have started training but not yet validated, most
    recently active first (by max(progress.csv, train.log) mtime).
    """
    candidates = []
    for job_dir in sweep_dir.glob("[0-9]*"):
        if not job_dir.is_dir() or not (job_dir / "train.log").exists():
            continue
        if (job_dir / "validation" / "summary.json").exists():
            continue
        mtime = _latest_mtime(job_dir, ("progress.csv", "train.log"))
        candidates.append((job_dir, mtime))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [job_dir for job_dir, _ in candidates[:max_jobs]]


def _build_in_progress_table(sweep_dir: Path) -> Table:
    table = Table(title="Currently training")
    table.add_column("Job")
    table.add_column("Fold")
    table.add_column("Epoch")
    table.add_column("Dice (ema)")
    table.add_column("Train loss")
    table.add_column("Val loss")
    table.add_column("s/epoch")
    table.add_column("Est. remaining")
    table.add_column("Last update")

    for job_dir in _find_in_progress_jobs(sweep_dir):
        progress = _current_epoch(job_dir)
        n_epoch, avg_sec = progress.n_epochs, progress.avg_sec
        config_path = job_dir / ".hydra" / "config.yaml"
        target, fold = None, None
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            target = (config.get("hp_config") or {}).get("num_epochs")
            fold = config.get("fold")

        if n_epoch is None:
            epoch_str = "starting..."
        elif target:
            epoch_str = f"{n_epoch}/{int(target)}"
        else:
            epoch_str = str(n_epoch)

        dice_str = f"{progress.dice:.4f}" if progress.dice is not None else "-"
        train_loss_str = f"{progress.train_loss:.4f}" if progress.train_loss is not None else "-"
        val_loss_str = f"{progress.val_loss:.4f}" if progress.val_loss is not None else "-"
        sec_str = f"{avg_sec:.0f}" if avg_sec is not None else "-"

        remaining_str = "n/a"
        if n_epoch is not None and target and avg_sec:
            remaining_epochs = max(int(target) - n_epoch, 0)
            remaining_str = _format_timedelta(remaining_epochs * avg_sec)

        mtime = _latest_mtime(job_dir, ("progress.csv", "train.log"))
        last_update_str = _format_ago(mtime) if mtime is not None else "?"
        table.add_row(
            job_dir.name, str(fold) if fold is not None else "?", epoch_str,
            dice_str, train_loss_str, val_loss_str, sec_str,
            remaining_str, last_update_str,
        )

    return table


def _hyperband_schedule(sweep_dir: Path) -> list[tuple[int, float]] | None:
    """[(n_configs, budget)] per rung slot (bracket s=0 - the schedule shape
    this project's tune_hpo*.yaml configs are built around), derived from
    min_budget/max_budget/eta on any already-completed job's resolved config.
    None if this sweep doesn't use that schedule shape, or nothing has
    completed yet to read a config from.
    """
    for job_dir in sorted(sweep_dir.glob("[0-9]*")):
        config_path = job_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if not all(k in config for k in ("min_budget", "max_budget", "eta")):
            return None
        from autonnunet.utils import compute_hyperband_budgets

        n_configs, budgets, *_ = compute_hyperband_budgets(
            b_min=config["min_budget"], b_max=config["max_budget"], eta=config["eta"],
            n_stages=1, sample_default_at_target=True, print_output=False,
        )
        return list(zip(n_configs[0], budgets[0]))
    return None


def _classify_job(job_dir: Path, rungs: list[tuple[int, float]]) -> tuple[int | None, list | None]:
    """(schedule slot, local_gpu_ids) for one completed job dir - reads
    config.yaml/overrides.yaml exactly once each (not twice, which the first
    version of this code did per job - see _hyperband_eta's docstring for why
    that mattered). Slot 0 is always the default config (config_id==0,
    identified via the `save=...config_0...` override), kept distinct from
    the identically-budgeted final promotion rung that would otherwise match
    the same budget value.
    """
    config_path = job_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None, None
    config = yaml.safe_load(config_path.read_text())
    gpu_ids = config.get("local_gpu_ids")
    num_epochs = (config.get("hp_config") or {}).get("num_epochs")
    overrides_path = job_dir / ".hydra" / "overrides.yaml"
    if num_epochs is None or not overrides_path.exists():
        return None, gpu_ids
    match = re.search(r"config_(\d+)", overrides_path.read_text())
    if match and int(match.group(1)) == 0:
        return 0, gpu_ids
    candidates = [(i, b) for i, (_, b) in enumerate(rungs) if i != 0]
    if not candidates:
        return None, gpu_ids
    slot, budget = min(candidates, key=lambda x: abs(x[1] - num_epochs))
    return (slot if abs(budget - num_epochs) < 0.5 else None), gpu_ids


def _cache_file_path(sweep_dir: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha256(str(sweep_dir.resolve()).encode()).hexdigest()[:16]
    return cache_dir / f"{digest}.json"


def _load_disk_cache(sweep_dir: Path, cache_dir: Path) -> dict:
    """Loads a persisted hb_cache from a previous monitor process, if one
    exists and still looks like it belongs to this exact sweep.

    Gated on `sweep_dir.stat().st_ctime`: this project has had job
    directories silently get reused for a different run twice before (a
    full sweep backup+restart recreating the whole directory, and separately
    a stale-job-directory bug from a mid-sweep config change) - a cache
    computed against the old contents would misreport an ETA for what's on
    disk now. If the sweep directory's own ctime doesn't match what was
    stored when the cache was written, the whole thing is discarded and
    rebuilt fresh (same one-time cost as having no cache, just not paid on
    every process start). A full backup+restart recreates sweep_dir itself,
    which this catches; a single job directory being silently reused without
    the whole sweep_dir changing is caught separately, per-job, inside
    _hyperband_eta (comparing each cached job's validation/summary.json
    mtime against what's on disk now).
    """
    path = _cache_file_path(sweep_dir, cache_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    try:
        if data.get("sweep_ctime") != sweep_dir.stat().st_ctime:
            return {}
    except OSError:
        return {}
    return {
        "schedule": data.get("schedule"),
        "concurrency": data.get("concurrency"),
        "jobs": {name: tuple(v) for name, v in data.get("jobs", {}).items()},
    }


def _save_disk_cache(sweep_dir: Path, cache_dir: Path, cache: dict) -> None:
    """Best-effort persist. Any failure (read-only mount, no permissions, a
    concurrent writer) just means the next process start pays the one-time
    scan cost again - never worth crashing a monitoring tool over.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sweep_ctime": sweep_dir.stat().st_ctime,
            "schedule": cache.get("schedule"),
            "concurrency": cache.get("concurrency"),
            "jobs": cache.get("jobs", {}),
        }
        path = _cache_file_path(sweep_dir, cache_dir)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.replace(path)
    except OSError:
        pass


def _hyperband_eta(sweep_dir: Path, cache: dict, cache_dir: Path | None = None) -> str | None:
    """Schedule-aware estimate of remaining wall-clock time: known rung
    structure (exact trial counts per fidelity, not just a total-budget
    fraction) x empirical per-rung timing, falling back for rungs with no
    completed samples yet to a training-rate + validation-floor extrapolation.

    That fallback is deliberately fit from the *promoted* rungs (slot >= 2)
    only, not the base rung: successive halving selectively promotes
    configs, and since this is a multi-objective search that explicitly
    rewards low runtime, the promoted population trends toward smaller/faster
    architectures than the full random draw at the base rung - their
    per-epoch rate is measurably different (validated empirically: a fit
    using the base rung overshoots the actually-observed promoted-rung times
    by 40-60%).

    `cache` persists across refreshes (see main()), and optionally across
    process restarts too if `cache_dir` is given (see _load_disk_cache /
    _save_disk_cache in main()). A completed job's data never changes, so
    once we've classified it we normally never touch its files again - only
    newly-completed job dirs get read on each call. Without this, re-parsing
    every job directory's YAML on every refresh - over a network-mounted
    output directory, with hundreds of job dirs and growing - makes each
    refresh take unacceptably long, which defeats the point of a monitoring
    tool.

    The one exception: for each already-cached job we still do one cheap
    `.stat()` on its validation/summary.json to confirm its mtime hasn't
    moved since we classified it. This project has twice had a job
    directory's contents silently change out from under an already-recorded
    result (a stale-job-directory reuse bug, and a full sweep backup+
    restart) - trusting a cached classification forever would let stale
    data linger in the ETA indefinitely in that scenario. The mtime check
    costs a single stat per cached job (no YAML parse), which is why it's
    cheap enough to do unconditionally rather than only mattering for the
    disk-cache path.

    Returns a formatted string, or None if there's not enough data yet to say
    anything better than the naive budget/wallclock estimate (e.g. a non-
    Hyperband sweep, or before the first rung has any completions).
    """
    schedule_just_computed = "schedule" not in cache
    if schedule_just_computed:
        cache["schedule"] = _hyperband_schedule(sweep_dir)
        cache["concurrency"] = None
        cache["jobs"] = {}  # job dir name -> (slot, duration_or_None, summary_mtime)

    rungs = cache["schedule"]
    if rungs is None:
        if schedule_just_computed and cache_dir is not None:
            _save_disk_cache(sweep_dir, cache_dir, cache)
        return None

    jobs_cache: dict[str, tuple] = cache["jobs"]
    dirty = schedule_just_computed
    for job_dir in sweep_dir.glob("[0-9]*"):
        name = job_dir.name
        summary_path = job_dir / "validation" / "summary.json"

        if name in jobs_cache:
            _, _, cached_mtime = jobs_cache[name]
            try:
                current_mtime = summary_path.stat().st_mtime
            except OSError:
                continue  # was complete before; a transient stat failure isn't worth reclassifying over
            if current_mtime == cached_mtime:
                continue  # unchanged since we classified it - trust the cache
            # summary.json's mtime moved since we cached this job: its result
            # changed underneath us (job directory reused for a different
            # run - has happened before in this project). Fall through and
            # re-classify rather than trust stale data.
        elif not summary_path.exists():
            continue  # not finished yet - cheap to recheck next refresh, don't cache

        slot, gpu_ids = _classify_job(job_dir, rungs)
        if cache["concurrency"] is None and gpu_ids:
            cache["concurrency"] = max(1, len(gpu_ids) // 3)  # 3-fold CV throughout this project

        duration = None
        summary_mtime = summary_path.stat().st_mtime
        debug_path = job_dir / "debug.json"
        if debug_path.exists():
            start = debug_path.stat().st_mtime
            if summary_mtime > start:
                duration = summary_mtime - start
        jobs_cache[name] = (slot, duration, summary_mtime)
        dirty = True

    if dirty and cache_dir is not None:
        _save_disk_cache(sweep_dir, cache_dir, cache)

    concurrency = cache["concurrency"] or 2

    slot_times: dict[int, list[float]] = defaultdict(list)
    slot_done: dict[int, int] = defaultdict(int)
    for slot, duration, _mtime in jobs_cache.values():
        if slot is None:
            continue
        slot_done[slot] += 1
        if duration is not None:
            slot_times[slot].append(duration)

    slot_means = {s: sum(v) / len(v) for s, v in slot_times.items() if v}
    if not slot_means:
        return None

    promoted_points = sorted((rungs[s][1], slot_means[s]) for s in slot_means if s >= 2)
    rate_per_epoch = validation_floor = None
    if len(promoted_points) >= 2:
        (e1, t1), (e2, t2) = promoted_points[0], promoted_points[-1]
        if e2 != e1:
            rate_per_epoch = (t2 - t1) / (e2 - e1)
            validation_floor = t1 - e1 * rate_per_epoch

    total_remaining_seconds = 0.0
    for slot, (width, budget) in enumerate(rungs):
        remaining = max(0, width - slot_done.get(slot, 0))
        if remaining == 0:
            continue
        if slot in slot_means:
            per_trial = slot_means[slot]
        elif rate_per_epoch is not None:
            per_trial = validation_floor + budget * rate_per_epoch
        else:
            return None  # a rung is missing and we have no basis to estimate it
        batches = -(-remaining // concurrency)  # ceil division
        total_remaining_seconds += batches * per_trial

    return _format_timedelta(total_remaining_seconds)


def _read_total_budget(sweep_dir: Path) -> tuple[int | None, int | None]:
    """Read `budget`/`n_trials` from any already-completed job's resolved Hydra config."""
    for job_dir in sorted(sweep_dir.glob("[0-9]*")):
        config_path = job_dir / ".hydra" / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text())
            return config.get("budget"), config.get("n_trials")
    return None, None


def _build_baseline_comparison(sweep_dir: Path, runhistory: pd.DataFrame | None) -> Panel | None:
    """Is the AutoML search actually on track to beat the default-config baseline?

    Everything needed (config_id, budget, o0_loss) is already in runhistory.csv,
    so - unlike the schedule-aware ETA - this needs no per-job-directory reads
    at all and is cheap on every refresh regardless of sweep size.

    The baseline is the default config (config_id == 0), evaluated once at the
    full target budget via `sample_default_at_target` - directly comparable to
    what the search's own final-rung configs will eventually reach. Anything
    still at a lower budget hasn't finished training, so its loss is a partial
    signal, not a final one: nnU-Net's validation loss is expected to only
    improve (or plateau) with more epochs, not get worse, so a partial-budget
    config already beating the baseline is a genuinely promising sign, but not
    proof it'll still be ahead once fully trained - and conversely, trailing at
    a low budget doesn't rule out catching up later. Framed accordingly below.

    Returns None if there isn't enough data yet (no runhistory, no baseline
    row, or no Hyperband schedule to break budgets into rungs).

    Each rung also shows a "baseline @ same epoch" figure: the baseline's own
    pseudo-Dice (`mean_fg_dice` in its progress.csv) after training for that
    same number of epochs - nnU-Net logs this every epoch for every job as a
    fast approximate validation, distinct from the full sliding-window
    inference used for the official Dice figures above it. It's a genuine
    same-epoch, apples-to-apples read of "how far along was the baseline
    itself at this point," at the cost of being a different (typically
    slightly lower/noisier) metric than the real validation Dice - so treat
    it as a trend signal alongside the headline numbers, not a third
    directly-comparable Dice value.
    """
    if runhistory is None or "config_id" not in runhistory.columns:
        return None
    loss_col = next((c for c in runhistory.columns if re.fullmatch(r"o\d+_loss", c)), None)
    if loss_col is None:
        return None

    baseline_rows = runhistory[runhistory["config_id"] == 0]
    if baseline_rows.empty:
        return None
    baseline_loss = float(baseline_rows[loss_col].iloc[0])
    baseline_dice = 1 - baseline_loss

    baseline_progress = _read_csv_safe(sweep_dir / "0" / "progress.csv")

    rungs = _hyperband_schedule(sweep_dir)
    rung_budgets = [b for _, b in rungs[1:]] if rungs is not None else sorted(runhistory["budget"].unique())

    search_rows = runhistory[runhistory["config_id"] != 0]
    lines = []
    best_overall: tuple[float, float] | None = None  # (budget, dice)
    final_tier_dice: float | None = None
    for i, budget in enumerate(rung_budgets):
        same_epoch_str = ""
        if baseline_progress is not None and "mean_fg_dice" in baseline_progress.columns:
            idx = min(max(int(round(budget)) - 1, 0), len(baseline_progress) - 1)
            baseline_pseudo_dice = float(baseline_progress["mean_fg_dice"].iloc[idx])
            same_epoch_str = f"      [dim]baseline @ same epoch: {baseline_pseudo_dice:.4f} pseudo-Dice[/dim]"

        tier = search_rows[(search_rows["budget"] - budget).abs() < 0.5]
        if tier.empty:
            lines.append(f"  rung {i + 1} ({budget:>6.0f} epochs): [dim]no completions yet[/dim]")
            if same_epoch_str:
                lines.append(same_epoch_str)
            continue
        best_row = tier.loc[tier[loss_col].idxmin()]
        best_dice = 1 - float(best_row[loss_col])
        delta = best_dice - baseline_dice
        verdict = "[green]ahead[/green]" if delta > 0 else "[yellow]behind[/yellow]"
        lines.append(
            f"  rung {i + 1} ({budget:>6.0f} epochs, {len(tier):>3d} done): "
            f"best Dice {best_dice:.4f} ({delta:+.4f} vs. baseline)  {verdict}  "
            f"[dim](config {int(best_row['config_id'])})[/dim]"
        )
        if same_epoch_str:
            lines.append(same_epoch_str)
        best_overall = (budget, best_dice)
        if budget == rung_budgets[-1]:
            final_tier_dice = best_dice

    if final_tier_dice is not None:
        delta = final_tier_dice - baseline_dice
        headline = (
            f"[bold green]Full-budget search config already beats the baseline by {delta:+.4f} Dice.[/bold green]"
            if delta > 0 else
            f"[bold yellow]Full-budget search config(s) still trail the baseline by {-delta:.4f} Dice.[/bold yellow]"
        )
    elif best_overall is not None:
        budget, dice = best_overall
        delta = dice - baseline_dice
        if delta > 0:
            headline = (
                f"[green]Promising:[/green] best config so far (at {budget:.0f}/{rung_budgets[-1]:.0f} "
                f"epochs) already beats the baseline by {delta:+.4f} Dice - not final, but a good sign."
            )
        else:
            headline = (
                f"[yellow]Not yet:[/yellow] best config so far (at {budget:.0f}/{rung_budgets[-1]:.0f} "
                f"epochs) still trails the baseline by {-delta:.4f} Dice, with promotions still to come."
            )
    else:
        headline = "[dim]No search configs completed yet - too early to say.[/dim]"

    body = "\n".join([
        f"Baseline (default config, full budget): Dice {baseline_dice:.4f}", "",
        headline, "", *lines,
    ])
    return Panel(body, title="AutoML vs. baseline")


def _format_timedelta(seconds: float) -> str:
    if seconds < 0:
        return "n/a"
    td = timedelta(seconds=int(seconds))
    days, rem = td.days, td.seconds
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _format_ago(dt: datetime) -> str:
    return _format_timedelta((datetime.now() - dt).total_seconds()) + " ago"


def _format_hp_value(value: object) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def build_overview(sweep_dir: Path, last_good_rows: dict, hb_cache: dict, cache_dir: Path | None = None) -> Group:
    """Assemble the renderable overview for one refresh.

    `last_good_rows` is mutated in place to cache the most recently successfully
    parsed row per incumbent file, so a transient mid-write read failure re-shows
    the last known-good data instead of blanking that row out.

    `hb_cache` is mutated in place by `_hyperband_eta` to avoid re-reading
    already-completed jobs' files on every refresh. If `cache_dir` is given,
    `hb_cache` is also persisted there so a freshly-started process doesn't
    have to pay the first-scan cost again either (see main()).
    """
    runhistory = _read_csv_safe(sweep_dir / "runhistory.csv")
    incumbent_files = sorted(sweep_dir.glob("incumbent_*.csv"))
    total_budget, n_trials = _read_total_budget(sweep_dir)

    last_update = _find_most_recent_activity(sweep_dir)
    for f in [sweep_dir / "runhistory.csv", *incumbent_files]:
        if f.exists():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if last_update is None or mtime > last_update:
                last_update = mtime

    n_configs = len(runhistory) if runhistory is not None else 0
    budget_used = None
    wallclock = None
    if incumbent_files:
        first = _read_csv_safe(incumbent_files[0])
        if first is not None:
            budget_used = float(first["budget_used"].iloc[-1])
            wallclock = float(first["total_wallclock_time"].iloc[-1])

    status_lines = [f"[bold]{sweep_dir}[/bold]"]
    if last_update is None:
        status_lines.append("[yellow]No sweep output found yet.[/yellow]")
    else:
        stale = datetime.now() - last_update > STALE_AFTER
        status = "[red]STALLED / NOT RUNNING[/red]" if stale else "[green]active[/green]"
        status_lines.append(f"Status: {status}  (last update {_format_ago(last_update)})")

    if n_trials:
        status_lines.append(f"Configs evaluated: {n_configs} / {n_trials} target trials")
    else:
        status_lines.append(f"Configs evaluated: {n_configs}")

    if budget_used is not None and total_budget:
        pct = 100 * budget_used / total_budget
        bar_width = 30
        filled = int(bar_width * min(pct, 100) / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        status_lines.append(f"Budget used: [{bar}] {budget_used:.0f} / {total_budget} ({pct:.1f}%)")
        if wallclock:
            hb_eta = _hyperband_eta(sweep_dir, hb_cache, cache_dir)
            if hb_eta is not None:
                eta_str = hb_eta
            elif budget_used > 0:
                # Fallback: naive linear extrapolation. Systematically too
                # pessimistic for a Hyperband schedule (the early, base-rung-
                # heavy phase pays a large fixed validation cost per trial
                # regardless of how little training it did, dragging the
                # average down) - only used when the schedule-aware estimate
                # above isn't available yet.
                rate = budget_used / wallclock
                remaining = max(total_budget - budget_used, 0)
                eta_seconds = remaining / rate if rate > 0 else -1
                eta_str = _format_timedelta(eta_seconds)
            else:
                eta_str = "n/a"
            status_lines.append(
                f"Wallclock so far: {_format_timedelta(wallclock)}   "
                f"Estimated remaining: {eta_str}"
            )

    status_panel = Panel("\n".join(status_lines), title="Sweep status")

    baseline_panel = _build_baseline_comparison(sweep_dir, runhistory)

    in_progress_table = _build_in_progress_table(sweep_dir)

    table = Table(title="Current incumbents")
    table.add_column("Objective")
    table.add_column("Value")
    table.add_column("run_id")
    table.add_column("Key hyperparameters")

    for f in incumbent_files:
        objective = f.stem.removeprefix("incumbent_")
        df = _read_csv_safe(f)
        if df is not None:
            value_col = next((c for c in df.columns if c.startswith("o") and c[1:].split("_")[0].isdigit()), None)
            last_row = df.iloc[-1]
            hp_cols = [c for c in df.columns if c.startswith("hp_config.")]
            hp_summary = ", ".join(
                f"{c.removeprefix('hp_config.')}={_format_hp_value(last_row[c])}" for c in hp_cols
            )
            value = f"{last_row[value_col]:.4f}" if value_col else "n/a"
            row = (value, str(int(last_row["run_id"])), hp_summary)
            last_good_rows[objective] = row
        elif objective in last_good_rows:
            row = last_good_rows[objective]
        else:
            continue
        table.add_row(objective, *row)

    renderables = [status_panel]
    if baseline_panel is not None:
        renderables.append(baseline_panel)
    renderables += [in_progress_table, table]
    return Group(*renderables)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--approach", type=str, default="hpo", choices=["hpo", "hpo_nas", "hpo_hnas"])
    argparser.add_argument("--dataset", type=str, required=True)
    argparser.add_argument("--configuration", type=str, default="3d_fullres")
    argparser.add_argument("--hpo_seed", type=int, default=0)
    argparser.add_argument("--interval", type=float, default=30, help="Refresh interval in seconds.")
    argparser.add_argument("--once", action="store_true", help="Print a single snapshot and exit.")
    argparser.add_argument(
        "--cache_dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Where to persist the schedule-aware ETA's per-job cache across process "
             "restarts (default: ~/.cache/autonnunet_monitor). Lets --once and freshly "
             "started watch sessions skip the one-time full-sweep scan instead of paying "
             "it on every launch. Safe to point at a throwaway/shared location: it's keyed "
             "by sweep_dir's own creation time and auto-discards itself if that sweep gets "
             "backed up and restarted from scratch.",
    )
    argparser.add_argument(
        "--no_disk_cache", action="store_true",
        help="Don't read or write the on-disk ETA cache; always recompute from scratch "
             "in memory for this process only.",
    )
    args = argparser.parse_args()

    sweep_dir = AUTONNUNET_OUTPUT / args.approach / args.dataset / args.configuration / str(args.hpo_seed)
    last_good_rows: dict = {}
    cache_dir = None if args.no_disk_cache else args.cache_dir
    hb_cache: dict = _load_disk_cache(sweep_dir, cache_dir) if cache_dir is not None else {}

    console = Console()
    if args.once:
        console.print(build_overview(sweep_dir, last_good_rows, hb_cache, cache_dir))
        return

    console.print("[dim]Press Ctrl+C to stop watching.[/dim]")

    # auto_refresh redraws in a background thread up to refresh_per_second times a
    # second regardless of whether content changed; since we only change content
    # once per --interval, that's pure redundant redraw and the likely flicker
    # source. Drive refreshes ourselves, once per update.
    try:
        with Live(
            build_overview(sweep_dir, last_good_rows, hb_cache, cache_dir),
            console=console, screen=False, auto_refresh=False,
        ) as live:
            while True:
                time.sleep(args.interval)
                live.update(build_overview(sweep_dir, last_good_rows, hb_cache, cache_dir), refresh=True)
    except KeyboardInterrupt:
        console.print("[dim]Stopped.[/dim]")


if __name__ == "__main__":
    main()
