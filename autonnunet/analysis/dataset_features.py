from autonnunet.utils.helpers import load_json, dataset_name_to_msd_task
from autonnunet.utils.paths import NNUNET_DATASETS, NNUNET_PREPROCESSED, AUTONNUNET_OUTPUT
from autonnunet.analysis.dataset_metrics import compute_compactness, compute_std_per_axis
from dataclasses import dataclass
from typing import Literal
import nibabel as nib
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

SOURCES = {
    "Dataset001_BrainTumour": "mp-MRI",
    "Dataset002_Heart": "MRI",
    "Dataset003_Liver": "CT",
    "Dataset004_Hippocampus": "MRI",
    "Dataset005_Prostate": "mp-MRI",
    "Dataset006_Lung": "CT",
    "Dataset007_Pancreas": "CT",
    "Dataset008_Spleen": "CT",
    "Dataset009_Spleen": "CT",
    "Dataset010_Colon": "CT",
}

def _load_original_dataset_info(dataset: str) -> dict:
    return load_json(NNUNET_DATASETS / dataset_name_to_msd_task(dataset_name=dataset) / "dataset.json")

def _load_nnunet_dataset_fingerprint(dataset: str) -> dict:
    return load_json(NNUNET_PREPROCESSED / dataset / "dataset_fingerprint.json")

def _load_cached_dataset_features(dataset: str) -> pd.DataFrame | None:
    path = AUTONNUNET_OUTPUT / "nnunet_dataset_features" / f"{dataset}.csv"
    if path.exists():
        return pd.read_csv(path)
    else:
        return None
    
def _cache_dataset_features(dataset: str, df: pd.DataFrame) -> None:
    path = AUTONNUNET_OUTPUT / "nnunet_dataset_features" / f"{dataset}.csv"
    if path.exists():
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _get_image_and_label_features(dataset: str) -> pd.DataFrame:
    dataset_info = _load_original_dataset_info(dataset)

    class_idxs = list(dataset_info["labels"].keys())
    train_samples = dataset_info["training"]

    rows = []

    base_path = NNUNET_DATASETS / dataset_name_to_msd_task(dataset_name=dataset)
    for sample in tqdm(train_samples):
        img_path, label_path = sample["image"], sample["label"]
        instance = img_path.split("/")[-1].replace(".nii.gz", "")

        img = nib.load(base_path / img_path).get_fdata()
        label = nib.load(base_path / label_path).get_fdata()
        shape = label.shape
        volume = np.prod(shape)

        for class_idx in class_idxs:
            class_mask = (label == float(class_idx))

            class_volume = class_mask.sum()
            compactness = compute_compactness(class_mask)
            std = compute_std_per_axis(class_mask)

            mean_intensity = img[class_mask].mean()
            std_intensity = img[class_mask].std()
            min_intensity = img[class_mask].min()
            max_intensity = img[class_mask].max()
            
            rows.append({
                "instance": instance,
                **{f"shape_{i}": shape[i] for i in range(3)},
                "volume": volume,
                "class_idx": class_idx,
                "class_volume": class_volume,
                "class_volume_ratio": class_volume / volume,
                "compactness": compactness,
                "std": std,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "min_intensity": min_intensity,
                "max_intensity": max_intensity,
            })

    return pd.DataFrame(rows)


def _get_dataset_features(dataset: str) -> dict:
    info = load_json(NNUNET_PREPROCESSED / dataset / "dataset.json")

    modality = int(info["tensorImageSize"][0])
    n_training_samples = int(info["numTraining"])
    n_test_samples = int(info["numTest"])
    n_channels = len(info["channel_names"])
    n_classes = len(info["labels"])
    source = SOURCES[dataset]

    return {
        "modality": modality,
        "n_training_samples": n_training_samples,
        "n_test_samples": n_test_samples,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "source": source,
    }


def extract_dataset_features(dataset: str, recompute: bool = False) -> pd.DataFrame:
    if not recompute and (dataset_features := _load_cached_dataset_features(dataset)) is not None:
        return dataset_features
    
    nnunet_dataset_features = _get_dataset_features(dataset)
    dataset_features = _get_image_and_label_features(dataset)
    for k, v in nnunet_dataset_features.items():
        dataset_features[k] = v
    
    _cache_dataset_features(dataset, dataset_features)

    return dataset_features


if __name__ == "__main__":
    nnunet_dataset_features = extract_dataset_features("Dataset004_Hippocampus", recompute=True)
    print(nnunet_dataset_features)
