# Based on https://github.com/bowang-lab/MedSAM/blob/MedSAM2/finetune_sam2_img.py
"""finetune sam2 model on medical image data
only finetune the image encoder and mask decoder
freeze the prompt encoder.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import os
import sys

sys.path.append(os.path.abspath("submodules/MedSAM"))
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

join = os.path.join
import datetime
import glob
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import hydra
import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from codecarbon import OfflineEmissionsTracker
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from autonnunet.utils.paths import MEDSAM2_PREPROCESSED

if TYPE_CHECKING:
    from omegaconf import DictConfig

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def get_bbox(mask, bbox_shift=5):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    return np.array([x_min, y_min, x_max, y_max])

@torch.no_grad()
def medsam_inference(
    medsam_model,
    features,
    box_1024,
    H, W
    ):
    img_embed, high_res_features = features["image_embed"], features["high_res_feats"]
    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=img_embed.device)
        box_labels = box_labels.repeat(box_torch.size(0), 1)
    concat_points = (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = medsam_model.sam2_model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )
    low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = medsam_model.sam2_model.sam_mask_decoder(
        image_embeddings=img_embed, # (1, 256, 64, 64)
        image_pe=medsam_model.sam2_model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    low_res_pred = torch.sigmoid(low_res_masks_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    return (low_res_pred > 0.5).astype(np.uint8)


def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
        the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.nan
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def compute_multi_class_dsc(gt, npz_seg) -> dict:
    dsc = {}
    labels = np.unique(gt)[1:]
    for i in labels:
        gt_i = gt == i
        seg_i = npz_seg[i] if i in npz_seg else np.zeros_like(gt_i)
        if np.sum(gt_i)==0 and np.sum(seg_i)==0:
            dsc[i] = 1
        elif np.sum(gt_i)==0 and np.sum(seg_i)>0:
            dsc[i] = 0
        else:
            dsc[i] = compute_dice_coefficient(gt_i, seg_i)

    return dsc

def validate(name: str, medsam_model, device, npz_path_tr, cfg) -> dict:
    sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

    _name = name.split(cfg.img_name_suffix)[0] + ".npz"

    if not os.path.exists(join(npz_path_tr, _name)):
        return {}

    npz = np.load(join(npz_path_tr, _name), allow_pickle=True)
    img_3D = npz["imgs"]

    segs_dict = {}
    gt_3D = npz["gts"]
    label_ids = np.unique(gt_3D)[1:]
    ## Simulate 3D box for each organ
    for label_id in label_ids:
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8)
        marker_data_id = (gt_3D == label_id).astype(np.uint8)
        marker_zids, _, _ = np.where(marker_data_id > 0)
        marker_zids = np.sort(np.unique(marker_zids))
        bbox_dict = {} # key: z_index, value: bbox
        for z in marker_zids:
            z_box = get_bbox(marker_data_id[z, :, :])
            bbox_dict[z] = z_box
        # find largest bbox in bbox_dict
        bbox_areas = [np.prod(bbox_dict[z][2:] - bbox_dict[z][:2]) for z in bbox_dict]
        z_max_area = list(bbox_dict.keys())[np.argmax(bbox_areas)]
        z_min = min(bbox_dict.keys())
        z_max = max(bbox_dict.keys())
        mid_slice_bbox_2d = bbox_dict[z_max_area]

        z_middle = int((z_max - z_min)/2 + z_min)

        z_max = min(z_max+1, img_3D.shape[0])
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            # convert the shape to (3, H, W)
            img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
            # get the image embedding
            with torch.no_grad():
                _features = medsam_model._image_encoder(img_1024_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                if np.max(pre_seg1024) > 0:
                    box_1024 = get_bbox(pre_seg1024)
                else:
                    box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None,:], H, W)
            segs_3d_temp[z, img_2d_seg>0] = 1

        # infer from middle slice to the z_min
        z_min = max(-1, z_min-1)
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
            # get the image embedding
            with torch.no_grad():
                _features = medsam_model._image_encoder(img_1024_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            if np.max(pre_seg1024) > 0:
                box_1024 = get_bbox(pre_seg1024)
            else:
                box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None,:], H, W)
            segs_3d_temp[z, img_2d_seg>0] = 1
        segs_dict[label_id] = segs_3d_temp.copy() ## save the segmentation result in one-hot format

    if not os.path.exists("validation"):
        os.makedirs("validation")

    for label_id in label_ids:
        seg_sitk = sitk.GetImageFromArray(segs_dict[label_id])
        seg_sitk.SetSpacing(npz["spacing"])
        sitk.WriteImage(seg_sitk, join("validation", _name.replace(".npz", f"_{label_id}.nii.gz")))

    dsc = compute_multi_class_dsc(gt_3D, segs_dict)
    dsc["case"] = name
    return dsc

@hydra.main(version_base=None, config_path="configs", config_name="finetune_medsam2")
def run(cfg: DictConfig):
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))

    from autonnunet.utils import seed_everything
    from autonnunet.utils.helpers import get_train_val_test_names

    logger = logging.getLogger()

    seed_everything(cfg.seed)

    logger.setLevel(logging.INFO)
    logger.info("Starting training script")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    class NpyDataset(Dataset):
        def __init__(self, data_root: str, names: list[str], bbox_shift=20):
            self.data_root = data_root
            self.gt_path = join(data_root, "gts")
            self.img_path = join(data_root, "imgs")
            self.gt_path_files = []
            for name in names:
                name = name.split(cfg.img_name_suffix)[0]
                self.gt_path_files.extend(
                    glob.glob(join(self.gt_path, name + "*"))
                )

            self.gt_path_files = self.gt_path_files
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if os.path.isfile(join(self.img_path, os.path.basename(file)))
            ]
            self.bbox_shift = bbox_shift
            self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
            logger.info(f"number of images: {len(self.gt_path_files)}")


        def __len__(self):
            return len(self.gt_path_files)

        def __getitem__(self, index):
            # load npy image (1024, 1024, 3), [0,1]
            img_name = os.path.basename(self.gt_path_files[index])
            img = np.load(
                join(self.img_path, img_name), "r", allow_pickle=True
            )  # (1024, 1024, 3)
            # convert the shape to (3, H, W)
            img_1024 = self._transform(img.copy())
            gt = np.load(
                self.gt_path_files[index], "r", allow_pickle=True
            )  # multiple labels [0, 1,4,5...], (256,256)
            assert img_name == os.path.basename(self.gt_path_files[index]), (
                "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
            )
            assert gt.shape == (256, 256), "ground truth should be 256x256"
            label_ids = np.unique(gt)[1:]
            gt2D = np.uint8(
                gt == random.choice(label_ids.tolist())
            )  # only one label, (256, 256)
            assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))

            bboxes = np.array([x_min, y_min, x_max, y_max])*4 ## scale bbox from 256 to 1024

            return (
                img_1024, ## [3, 1024, 1024]
                torch.tensor(gt2D[None, :, :]).long(), ## [1, 256, 256]
                torch.tensor(bboxes).float(),
                img_name,
            )

    class MedSAM2(nn.Module):
        def __init__(
            self,
            model,
        ):
            super().__init__()
            self.sam2_model = model
            # freeze prompt encoder
            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False


        def forward(self, image, box):
            """image: (B, 3, 1024, 1024)
            box: (B, 2, 2).
            """
            _features = self._image_encoder(image)
            img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
            # do not compute gradients for prompt encoder
            with torch.no_grad():
                box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                    box_labels = box_labels.repeat(box_torch.size(0), 1)
                concat_points = (box_coords, box_labels)

                sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                    points=concat_points,
                    boxes=None,
                    masks=None,
                )
            low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
                image_embeddings=img_embed, # (B, 256, 64, 64)
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            return low_res_masks_logits

        def _image_encoder(self, input_image):
            backbone_out = self.sam2_model.forward_image(input_image)
            _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
            # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
            if self.sam2_model.directly_add_no_mem_embed:
                vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
            bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
            feats = [
                feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1], strict=False)
            ][::-1]
            return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


    device = torch.device(cfg.device)
    # Get path of current file
    file_path = Path(__file__).parent.parent.resolve()
    sam2_checkpoint = file_path / "submodules" / "MedSAM" / "checkpoints" / cfg.pretrained_model
    logger.info("Loading SAM2 model")
    sam2_model = build_sam2(model_cfg=cfg.model, ckpt_path=str(sam2_checkpoint), device=device)
    medsam_model = MedSAM2(model=sam2_model)
    medsam_model.train()

    logger.info(
        f"Number of total parameters: {sum(p.numel() for p in medsam_model.parameters())}"
    )
    logger.info(
        f"Number of trainable parameters: {sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)}"
    )

    img_mask_encdec_params = list(medsam_model.sam2_model.image_encoder.parameters()) + list(
        medsam_model.sam2_model.sam_mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    logger.info(
        f"Number of image encoder and mask decoder parameters: {sum(p.numel() for p in img_mask_encdec_params if p.requires_grad)}"
    )
    seg_loss_func = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    fg_seg_loss_func = monai.losses.DiceLoss(include_background=False, sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    npy_path_tr = MEDSAM2_PREPROCESSED / cfg.dataset.name / "imagesTr" / "npy"
    npz_path_tr = MEDSAM2_PREPROCESSED / cfg.dataset.name / "imagesTr" / "npz"

    train_names, val_names, _ = get_train_val_test_names(cfg)

    train_dataset = NpyDataset(str(npy_path_tr), train_names, bbox_shift=cfg.bbox_shift)
    val_dataset = NpyDataset(str(npy_path_tr), val_names, bbox_shift=cfg.bbox_shift)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_train_workers,
        pin_memory=True,
    )

    logger.info(f"Number of validation samples: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_val_workers,
        pin_memory=True,
    )

    num_epochs = cfg.num_epochs
    train_seg_losses = []
    train_ce_losses = []
    val_seg_losses = []
    val_fg_seg_losses = []
    val_ce_losses = []
    epoch_runtimes = []
    best_val_fg_seg_loss = 1e10
    start_epoch = 0

    if cfg.resume:
        if os.path.isfile("medsam_model_latest.pth") and os.path.isfile("progress.csv"):
            ## Map model to be loaded to specified single GPU
            logger.info("=> loading checkpoint 'medsam_model_latest.pth'")
            checkpoint = torch.load("medsam_model_latest.pth", map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])

            progress = pd.read_csv("progress.csv")
            train_seg_losses = progress["Training Segmentation Loss"].tolist()
            train_ce_losses = progress["Training CrossEntropy Loss"].tolist()
            val_seg_losses = progress["Validation Segmentation Loss"].tolist()
            val_fg_seg_losses = progress["Validation Foreground Segmentation Loss"].tolist()
            val_ce_losses = progress["Validation CrossEntropy Loss"].tolist()
            epoch_runtimes = progress["Epoch Runtime"].tolist()
            logger.info(f"Loaded checkpoint 'medsam_model_latest.pth' (epoch {start_epoch})")
        else:
            logger.info("=> no checkpoint found at 'medsam_model_latest.pth'")

    training = num_epochs - start_epoch > 0

    if training:
        logger.info("Setting up emissions tracker")
        if os.path.exists("emissions.csv"):
            os.rename("emissions.csv", "emissions_old.csv")

        tracker = OfflineEmissionsTracker(
            country_iso_code="DEU",
            log_level="WARNING"
        )
        tracker.start()

    logger.info(f"Running training for {num_epochs - start_epoch} epochs")
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_train_seg_loss = 0
        epoch_train_ce_loss = 0

        medsam_model.train()
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            medsam_pred = medsam_model(image, boxes_np)
            seg_loss = seg_loss_func(medsam_pred, gt2D)
            ce_loss = ce_loss_func(medsam_pred, gt2D.float())
            loss = seg_loss + ce_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_seg_loss += seg_loss.item()
            epoch_train_ce_loss += ce_loss.item()

        epoch_train_seg_loss /= step
        epoch_train_ce_loss /= step

        train_seg_losses.append(epoch_train_seg_loss)
        train_ce_losses.append(epoch_train_ce_loss)

        medsam_model.eval()
        epoch_val_fg_seg_loss = 0
        epoch_val_seg_loss = 0
        epoch_val_ce_loss = 0

        with torch.no_grad():
            for step, (image, gt2D, boxes, _) in enumerate(tqdm(val_dataloader)):
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                medsam_pred = medsam_model(image, boxes_np)

                # To compare to nnU-Net we use foreground dice loss
                fg_seg_loss = fg_seg_loss_func(medsam_pred, gt2D)
                seg_loss = seg_loss_func(medsam_pred, gt2D)

                ce_loss = ce_loss_func(medsam_pred, gt2D.float())
                loss = seg_loss + ce_loss

                epoch_val_fg_seg_loss = fg_seg_loss.item()
                epoch_val_seg_loss += seg_loss.item()
                epoch_val_ce_loss += ce_loss.item()

        epoch_val_fg_seg_loss /= step
        epoch_val_seg_loss /= step
        epoch_val_ce_loss /= step

        val_fg_seg_losses.append(epoch_val_fg_seg_loss)
        val_seg_losses.append(epoch_val_seg_loss)
        val_ce_losses.append(epoch_val_ce_loss)
        epoch_runtime = time.time() - epoch_start
        epoch_runtimes.append(epoch_runtime)

        logger.info(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train Seg. Loss: {epoch_train_seg_loss},Train CE Loss: {epoch_train_ce_loss}, Val Seg. Loss: {epoch_val_seg_loss}, Val CE Loss: {epoch_val_ce_loss}'
        )

        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, "medsam_model_latest.pth")

        ## save the best model
        if epoch_val_seg_loss < best_val_fg_seg_loss:
            best_val_fg_seg_loss = epoch_val_seg_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, "medsam_model_best.pth")

        progress = pd.DataFrame({
            "Epoch": np.arange(len(train_seg_losses)),
            "Training Segmentation Loss": train_seg_losses,
            "Training CrossEntropy Loss": train_ce_losses,
            "Validation Segmentation Loss": val_seg_losses,
            "Validation Foreground Segmentation Loss": val_fg_seg_losses,
            "Validation CrossEntropy Loss": val_ce_losses,
            "Epoch Runtime": epoch_runtimes,
        })
        progress.to_csv("progress.csv", index=False)

    if training:
        tracker.stop()

        if os.path.exists("emissions_old.csv"):
            emissions_old = pd.read_csv("emissions_old.csv")
            emissions_new = pd.read_csv("emissions.csv")
            emissions = pd.concat([emissions_old, emissions_new])
            emissions.to_csv("emissions.csv", index=False)
            os.remove("emissions_old.csv")

    logger.info("Training done")

    # Perform actual validation
    logger.info("Starting validation")
    shutil.rmtree("validation", ignore_errors=True)
    logger.info("Loading best model checkpoint")
    best_medsam2_checkpoint = torch.load("medsam_model_best.pth", map_location=device)
    medsam_model.load_state_dict(best_medsam2_checkpoint["model"], strict=True)
    medsam_model.eval()
    logger.info("Checkpoint loaded")

    logger.info("Starting validation")
    results = []
    for name in tqdm(val_names):
        result = validate(name, medsam_model, device, npz_path_tr, cfg)
        if len(result) > 0:
            results.append(result)

    result_df = pd.DataFrame(results)
    result_df.to_csv("validation_results.csv", index=False)
    logger.info("Validation done")

if __name__  == "__main__":
    sys.exit(run())