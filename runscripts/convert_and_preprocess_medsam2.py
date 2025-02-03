"""Convert and preprocess a dataset for MedSAM2."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import SimpleITK as sitk
import os

join = os.path.join
from tqdm import tqdm
import cc3d
import sys
import multiprocessing as mp
from functools import partial
import cv2
from typing import TYPE_CHECKING

import hydra
from autonnunet.utils.paths import NNUNET_RAW, MEDSAM2_PREPROCESSED
from autonnunet.utils.helpers import get_train_val_test_names

if TYPE_CHECKING:
    from omegaconf import DictConfig


def preprocess(name: str, npy_path: str, npz_path: str, cfg: DictConfig):
    """
    Preprocess the image and ground truth, and save them as npz files

    Parameters
    ----------
    name : str
        name of the ground truth file
    npz_path : str
        path to save the npz files
    """
    img_path = str(NNUNET_RAW / cfg.dataset.name / "imagesTr")
    gt_path = str(NNUNET_RAW / cfg.dataset.name / "labelsTr")

    image_name = name.split(cfg.img_name_suffix)[0] + cfg.img_name_suffix

    # We need to remove the modality suffix, e.g. "_0000" 
    gt_name = name.split(cfg.gt_name_suffix)[0][:-5] + cfg.gt_name_suffix

    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data = np.uint8(sitk.GetArrayFromImage(gt_sitk))

    gt_data_ori = gt_data.copy()

    # exclude the objects with less than 1000 pixels in 3D
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=cfg.voxel_num_thre3d, connectivity=26, in_place=True
    )

    # remove small objects with less than 100 pixels in 2D slices
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=cfg.voxel_num_thre2d, connectivity=8, in_place=True
        )

    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(img_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if cfg.modality == "CT":
            lower_bound = cfg.window_level - cfg.window_width / 2
            upper_bound = cfg.window_level + cfg.window_width / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]

        # we use the original preprocessed image for the validation 
        np.savez_compressed(join(npz_path, image_name.split(cfg.img_name_suffix)[0] + ".npz"), imgs=image_data_pre, gts=gt_data, spacing=img_sitk.GetSpacing())

        # Save as npy files
        name = name.split(cfg.img_name_suffix)[0]
        if len(gt_roi.shape) > 2: ## 3D image
            for i in range(img_roi.shape[0]):
                img_i = img_roi[i, :, :]
                img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                gt_i = gt_roi[i, :, :]
                gt_i = np.uint8(gt_i)
                gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
                assert gt_i.shape == (256, 256)
                np.save(join(npy_path, "imgs", name + "-" + str(i).zfill(3) + ".npy"), img_3c)
                np.save(join(npy_path, "gts", name + "-" + str(i).zfill(3) + ".npy"), gt_i)
        else: ## 2D image
            if len(img_roi.shape) < 3:
                img_3c = np.repeat(img_roi[:, :, None], 3, axis=-1)
            else:
                img_3c = img_roi

            gt_i = gt_roi
            gt_i = np.uint8(gt_i)
            gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
            assert gt_i.shape == (256, 256)
            np.save(join(npy_path, "imgs", name + ".npy"), img_3c)
            np.save(join(npy_path, "gts", name + ".npy"), gt_i)


@hydra.main(version_base=None, config_path="configs", config_name="convert_and_preprocess_medsam2")
def run(cfg: DictConfig):    
    names_train, names_val, _ = get_train_val_test_names(cfg)
    names_train_val = names_train + names_val

    npz_path_tr = MEDSAM2_PREPROCESSED / cfg.dataset.name / "imagesTr" / "npz"
    npz_path_tr.mkdir(parents=True, exist_ok=True)
    npy_path_tr = MEDSAM2_PREPROCESSED / cfg.dataset.name /  "imagesTr" / "npy"
    (npy_path_tr / "imgs").mkdir(parents=True, exist_ok=True)
    (npy_path_tr / "gts").mkdir(parents=True, exist_ok=True)

    preprocess_tr = partial(preprocess, npy_path=str(npy_path_tr), npz_path=str(npz_path_tr), cfg=cfg)

    with mp.Pool(cfg.num_workers) as p:
        with tqdm(total=len(names_train_val)) as pbar:
            pbar.set_description("Preprocessing training images")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, names_train_val))):
                pbar.update()


if __name__  == "__main__":
    sys.exit(run())
    