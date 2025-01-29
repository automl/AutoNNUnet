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
from autonnunet.utils.paths import NNUNET_RAW, NNUNET_PREPROCESSED, MEDSAM2_PREPROCESSED
from autonnunet.utils.helpers import dataset_name_to_msd_task, load_json

if TYPE_CHECKING:
    from omegaconf import DictConfig


def preprocess(name: str, npz_path: str, cfg: DictConfig):
    """
    Preprocess the image and ground truth, and save them as npz files

    Parameters
    ----------
    name : str
        name of the ground truth file
    npz_path : str
        path to save the npz files
    """
    prefix = cfg.dataset.name + "_"
    img_path = str(NNUNET_RAW / cfg.dataset.name / "imagesTr")
    gt_path = str(NNUNET_RAW / cfg.dataset.name / "labelsTr")

    image_name = name.split(cfg.img_name_suffix)[0] + cfg.img_name_suffix

    # We need to remove the modality suffix, e.g. "_0000" 
    gt_name = name.split(cfg.gt_name_suffix)[0][:-5] + cfg.gt_name_suffix

    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

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
        np.savez_compressed(join(npz_path, gt_name.split(cfg.gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())

        # save the image and ground truth as nii files for sanity check;
        # they can be removed
        if cfg.save_nii:
            img_roi_sitk = sitk.GetImageFromArray(img_roi)
            img_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(
                img_roi_sitk,
                join(npz_path, gt_name.split(cfg.gt_name_suffix)[0] + "_img.nii.gz"),
            )
            gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
            gt_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(
                gt_roi_sitk,
                join(npz_path, prefix + gt_name.split(cfg.gt_name_suffix)[0] + "_gt.nii.gz"),
            )

def convert_npz_to_npy(npz_name: str, npz_dir: str, npy_dir: str):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_name : str
        Name of the npz file to be converted
    """
    name = npz_name.split(".npz")[0]
    npz_path = join(npz_dir, npz_name)
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"]
    gts = npz["gts"]
    if len(gts.shape) > 2: ## 3D image
        for i in range(imgs.shape[0]):
            img_i = imgs[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            gt_i = gts[i, :, :]
            gt_i = np.uint8(gt_i)
            gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
            assert gt_i.shape == (256, 256)
            np.save(join(npy_dir, "imgs", name + "-" + str(i).zfill(3) + ".npy"), img_3c)
            np.save(join(npy_dir, "gts", name + "-" + str(i).zfill(3) + ".npy"), gt_i)
    else: ## 2D image
        if len(imgs.shape) < 3:
            img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
        else:
            img_3c = imgs

        gt_i = gts
        gt_i = np.uint8(gt_i)
        gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
        assert gt_i.shape == (256, 256)
        np.save(join(npy_dir, "imgs", name + ".npy"), img_3c)
        np.save(join(npy_dir, "gts", name + ".npy"), gt_i)

def get_train_val_names(cfg: DictConfig) -> tuple[list[str], list[str]]:
    preprocessed_folder = NNUNET_PREPROCESSED / cfg.dataset.name
    dataset_info = load_json(preprocessed_folder / "dataset.json")
    n_channels = len(dataset_info["channel_names"])
    splits = load_json(preprocessed_folder / "splits_final.json")

    train_imgs = []
    val_imgs = []
    for name in splits[cfg.fold]["train"]:
        for i in range(n_channels):
            train_imgs += [f"{name}_{'%04d' % i}.nii.gz"]
        
    for name in splits[cfg.fold]["val"]:
        for i in range(n_channels):
            val_imgs += [f"{name}_{'%04d' % i}.nii.gz"]

    return train_imgs, val_imgs


@hydra.main(version_base=None, config_path="configs", config_name="convert_and_preprocess_medsam2")
def run(cfg: DictConfig):
    tr_names, ts_names = get_train_val_names(cfg)
    output_path = MEDSAM2_PREPROCESSED / cfg.dataset.name / str(cfg.fold)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_train_path = output_path / "npz_train"
    npz_val_path = output_path / "npz_val"

    npz_train_path.mkdir(parents=True, exist_ok=True)
    npz_val_path.mkdir(parents=True, exist_ok=True)

    preprocess_tr = partial(preprocess, npz_path=str(npz_train_path), cfg=cfg)
    preprocess_ts = partial(preprocess, npz_path=str(npz_val_path), cfg=cfg)

    with mp.Pool(cfg.num_workers) as p:
        with tqdm(total=len(tr_names)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
                pbar.update()
        with tqdm(total=len(ts_names)) as pbar:
            pbar.set_description("Preprocessing testing data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_ts, ts_names))):
                pbar.update()

    npy_path_tr = output_path / "npy_train"
    npy_path_val = output_path / "npy_val"
    (npy_path_tr / "imgs").mkdir(parents=True, exist_ok=True)
    (npy_path_tr / "gts").mkdir(parents=True, exist_ok=True)
    (npy_path_val / "imgs").mkdir(parents=True, exist_ok=True)
    (npy_path_val / "gts").mkdir(parents=True, exist_ok=True)
                             
    npz_names_tr = [f for f in os.listdir(npz_train_path) if f.endswith(".npz")]
    npz_names_val = [f for f in os.listdir(npz_val_path) if f.endswith(".npz")]

    convert_npz_to_npy_tr = partial(convert_npz_to_npy, npz_dir=str(npz_train_path), npy_dir=str(npy_path_tr))
    convert_npz_to_npy_val = partial(convert_npz_to_npy, npz_dir=str(npz_val_path), npy_dir=str(npy_path_val))

    with mp.Pool(cfg.num_workers) as pool:
        with tqdm(total=len(npz_names_tr)) as pbar:
            pbar.set_description("Converting npz to npy")
            for i, _ in enumerate(pool.imap_unordered(convert_npz_to_npy_tr, npz_names_tr)):
                pbar.update()

    with mp.Pool(cfg.num_workers) as pool:
        with tqdm(total=len(npz_names_tr)) as pbar:
            pbar.set_description("Converting npz to npy")
            for i, _ in enumerate(pool.imap_unordered(convert_npz_to_npy_val, npz_names_val)):
                pbar.update()

if __name__  == "__main__":
    test = "la_005_0000.nii.gz"
    print()

    sys.exit(run())
    