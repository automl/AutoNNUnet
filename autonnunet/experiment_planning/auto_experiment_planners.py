# Taken from nnunetv2
from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
from dynamic_network_architectures.architectures.unet import \
    ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op, get_matching_instancenorm)
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import \
    ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import \
    get_pool_and_conv_props
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
    nnUNetPlannerResEncL, nnUNetPlannerResEncM, nnUNetPlannerResEncXL)
from nnunetv2.preprocessing.resampling.default_resampling import \
    compute_new_shape


class AutoExperimentPlanner(ExperimentPlanner):
    def plan_experiment(self):
        """MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY.

        Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
        differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
        surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

        So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
        hard.
        """
        # we use this as a cache to prevent having to instantiate the architecture too often. Saves computation time
        _tmp = {}

        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint["spacings"], self.dataset_fingerprint["shapes_after_crop"], strict=False)]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json["numTraining"])
        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier("3d_fullres"),
                                                               approximate_n_voxels_dataset, _tmp)
            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres["patch_size"]
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres["spacing"])

            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!
            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
                else:
                    lowres_spacing *= spacing_increase_factor
                median_num_voxels = np.prod(plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)
                # print(lowres_spacing)
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  tuple([round(i) for i in plan_3d_fullres["spacing"] /
                                                                         lowres_spacing * new_median_shape_transposed]),
                                                                  self.generate_data_identifier("3d_lowres"),
                                                                  float(np.prod(median_num_voxels) *
                                                                        self.dataset_json["numTraining"]), _tmp)
                num_voxels_in_patch = np.prod(plan_3d_lowres["patch_size"], dtype=np.int64)
                print(f'Attempting to find 3d_lowres config. '
                      f'\nCurrent spacing: {lowres_spacing}. '
                      f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                      f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')
            if np.prod(new_median_shape_transposed, dtype=np.float64) / median_num_voxels < 2:
                print(f'Dropping 3d_lowres config because the image size difference to 3d_fullres is too small. '
                      f'3d_fullres: {new_median_shape_transposed}, '
                      f'3d_lowres: {[round(i) for i in plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed]}')
                plan_3d_lowres = None
            if plan_3d_lowres is not None:
                plan_3d_lowres["batch_dice"] = False
                plan_3d_fullres["batch_dice"] = True
            else:
                plan_3d_fullres["batch_dice"] = False
        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # 2D configuration
        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                   new_median_shape_transposed[1:],
                                                   self.generate_data_identifier("2d"), approximate_n_voxels_dataset,
                                                   _tmp)
        plan_2d["batch_dice"] = True

        print("2D U-Net configuration:")
        print(plan_2d)
        print()

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint["spacings"], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint["shapes_after_crop"], 0)[transpose_forward]

        # json is ###. I hate it... "Object of type int64 is not JSON serializable"
        plans = {
            "dataset_name": self.dataset_name,
            "plans_name": self.plans_identifier,
            "original_median_spacing_after_transp": [float(i) for i in median_spacing],
            "original_median_shape_after_transp": [int(round(i)) for i in median_shape],
            "image_reader_writer": self.determine_reader_writer().__name__,
            "transpose_forward": [int(i) for i in transpose_forward],
            "transpose_backward": [int(i) for i in transpose_backward],
            "configurations": {"2d": plan_2d},
            "experiment_planner_used": self.__class__.__name__,
            "label_manager": "LabelManager",
            "foreground_intensity_properties_per_channel": self.dataset_fingerprint[
                "foreground_intensity_properties_per_channel"]
        }

        if plan_3d_lowres is not None:
            plans["configurations"]["3d_lowres"] = plan_3d_lowres
            if plan_3d_fullres is not None:
                plans["configurations"]["3d_lowres"]["next_stage"] = "3d_cascade_fullres"
            print("3D lowres U-Net configuration:")
            print(plan_3d_lowres)
            print()
        if plan_3d_fullres is not None:
            plans["configurations"]["3d_fullres"] = plan_3d_fullres
            print("3D fullres U-Net configuration:")
            print(plan_3d_fullres)
            print()
            if plan_3d_lowres is not None:
                plans["configurations"]["3d_cascade_fullres"] = {
                    "inherits_from": "3d_fullres",
                    "previous_stage": "3d_lowres"
                }

        return plans

class AutoResEncUNetPlanner(AutoExperimentPlanner):
    def __init__(self, dataset_name_or_id: str | int,
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = "DefaultPreprocessor", plans_name: str = "nnUNetResEncUNetPlans",
                 overwrite_target_spacing: list[float] | tuple[float, ...] | None = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet
        # the following two numbers are really arbitrary and were set to reproduce default nnU-Net's configurations as
        # much as possible
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.UNet_blocks_per_stage_encoder = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
        self.UNet_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """Configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from.
        """
        if configuration_name in ("2d", "3d_fullres"):
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return "nnUNetPlans" + "_" + configuration_name
        return self.plans_identifier + "_" + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: np.ndarray | tuple[float, ...] | list[float],
                                    median_shape: np.ndarray | tuple[int, ...],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        def _features_per_stage(num_stages, max_num_features) -> tuple[int, ...]:
            return tuple([min(max_num_features, self.UNet_base_num_features * 2 ** i) for
                          i in range(num_stages)])

        def _keygen(patch_size, strides):
            return str(patch_size) + "_" + str(strides)

        assert all(i > 0 for i in spacing), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json["channel_names"].keys()
                                 if "channel_names" in self.dataset_json
                                 else self.dataset_json["modality"].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d
        unet_conv_op = convert_dim_to_conv_op(len(spacing))

        # print(spacing, median_shape, approximate_n_voxels_dataset)
        # find an initial patch size
        # we first use the spacing to get an aspect ratio
        tmp = 1 / np.array(spacing)

        # we then upscale it so that it initially is certainly larger than what we need (rescale to have the same
        # volume as a patch of size 256 ** 3)
        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
        # ideal because large initial patch sizes increase computation time because more iterations in the while loop
        # further down may be required.
        if len(spacing) == 3:
            initial_patch_size = [round(i) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [round(i) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        # clip initial patch size to median_shape. It makes little sense to have it be larger than that. Note that
        # this is different from how nnU-Net v1 does it!
        # todo patch size can still get too large because we pad the patch size to a multiple of 2**n
        initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[:len(spacing)], strict=False)])

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = len(pool_op_kernel_sizes)

        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            "network_class_name": self.UNet_class.__module__ + "." + self.UNet_class.__name__,
            "arch_kwargs": {
                "n_stages": num_stages,
                "features_per_stage": _features_per_stage(num_stages, max_num_features),
                "conv_op": unet_conv_op.__module__ + "." + unet_conv_op.__name__,
                "kernel_sizes": conv_kernel_sizes,
                "strides": pool_op_kernel_sizes,
                "n_blocks_per_stage": self.UNet_blocks_per_stage_encoder[:num_stages],
                "n_conv_per_stage_decoder": self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                "conv_bias": True,
                "norm_op": norm.__module__ + "." + norm.__name__,
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            "_kw_requires_import": ("conv_op", "norm_op", "dropout_op", "nonlin"),
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache:
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json["labels"].keys()),
                                                       architecture_kwargs["network_class_name"],
                                                       architecture_kwargs["arch_kwargs"],
                                                       architecture_kwargs["_kw_requires_import"],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            # print(patch_size)
            # patch size seems to be too large, so we need to reduce it. Reduce the axis that currently violates the
            # aspect ratio the most (that is the largest relative to median shape)
            axis_to_be_reduced = np.argsort([i / j for i, j in zip(patch_size, median_shape[:len(spacing)], strict=False)])[-1]

            # we cannot simply reduce that axis by shape_must_be_divisible_by[axis_to_be_reduced] because this
            # may cause us to skip some valid sizes, for example shape_must_be_divisible_by is 64 for a shape of 256.
            # If we subtracted that we would end up with 192, skipping 224 which is also a valid patch size
            # (224 / 2**5 = 7; 7 < 2 * self.UNet_featuremap_min_edge_length(4) so it's valid). So we need to first
            # subtract shape_must_be_divisible_by, then recompute it and then subtract the
            # recomputed shape_must_be_divisible_by. Annoying.
            patch_size = list(patch_size)
            tmp = deepcopy(patch_size)
            tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]
            _, _, _, _, shape_must_be_divisible_by = \
                get_pool_and_conv_props(spacing, tmp,
                                        self.UNet_featuremap_min_edge_length,
                                        999999)
            patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

            # now recompute topology
            network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
            shape_must_be_divisible_by = get_pool_and_conv_props(spacing, patch_size,
                                                                 self.UNet_featuremap_min_edge_length,
                                                                 999999)

            num_stages = len(pool_op_kernel_sizes)
            architecture_kwargs["arch_kwargs"].update({
                "n_stages": num_stages,
                "kernel_sizes": conv_kernel_sizes,
                "strides": pool_op_kernel_sizes,
                "features_per_stage": _features_per_stage(num_stages, max_num_features),
                "n_blocks_per_stage": self.UNet_blocks_per_stage_encoder[:num_stages],
                "n_conv_per_stage_decoder": self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache:
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json["labels"].keys()),
                    architecture_kwargs["network_class_name"],
                    architecture_kwargs["arch_kwargs"],
                    architecture_kwargs["_kw_requires_import"],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        batch_size = round((reference / estimate) * ref_bs)

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
        batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        return {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
            "architecture": architecture_kwargs
        }


class nnUNetPlannerResEncM(AutoResEncUNetPlanner):
    """Target is ~9-11 GB VRAM max -> older Titan, RTX 2080ti."""
    def __init__(self, dataset_name_or_id: str | int,
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = "DefaultPreprocessor", plans_name: str = "nnUNetResEncUNetMPlans",
                 overwrite_target_spacing: list[float] | tuple[float, ...] | None = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 8:
            warnings.warn("WARNING: You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 8, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 8

        # this is supposed to give the same GPU memory requirement as the default nnU-Net
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.max_dataset_covered = 1


class nnUNetPlannerResEncL(AutoResEncUNetPlanner):
    """Target is ~24 GB VRAM max -> RTX 4090, Titan RTX, Quadro 6000."""
    def __init__(self, dataset_name_or_id: str | int,
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = "DefaultPreprocessor", plans_name: str = "nnUNetResEncUNetLPlans",
                 overwrite_target_spacing: list[float] | tuple[float, ...] | None = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 24:
            warnings.warn("WARNING: You are running nnUNetPlannerL with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 24, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 24

        self.UNet_reference_val_3d = 2100000000  # 1840000000
        self.UNet_reference_val_2d = 380000000  # 352666667
        self.max_dataset_covered = 1


class nnUNetPlannerResEncXL(AutoResEncUNetPlanner):
    """Target is 40 GB VRAM max -> A100 40GB, RTX 6000 Ada Generation."""
    def __init__(self, dataset_name_or_id: str | int,
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = "DefaultPreprocessor", plans_name: str = "nnUNetResEncUNetXLPlans",
                 overwrite_target_spacing: list[float] | tuple[float, ...] | None = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 40:
            warnings.warn("WARNING: You are running nnUNetPlannerXL with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 40, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 40

        self.UNet_reference_val_3d = 3600000000
        self.UNet_reference_val_2d = 560000000
        self.max_dataset_covered = 1

