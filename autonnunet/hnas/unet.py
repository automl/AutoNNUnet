from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union, List, Tuple
import re
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He, init_last_bn_before_add_to_0)
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim, convert_dim_to_conv_op, get_matching_batchnorm,
    get_matching_dropout, get_matching_instancenorm)
from torch import nn
import logging
import pydoc

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger("CFGUNet")

class CFGUNet(nn.Module):
    def __init__(
            self,
            hp_config: DictConfig,
            string_tree: str,
            arch_init_kwargs: dict,
            arch_kwargs_req_import: List[str],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool
        ):
        super().__init__()

        self.hp_config = hp_config
        self.string_tree = string_tree

        self.architecture_kwargs = dict(**arch_init_kwargs)   
        for ri in arch_kwargs_req_import:
            if self.architecture_kwargs[ri] is not None:
                self.architecture_kwargs[ri] = pydoc.locate(self.architecture_kwargs[ri])

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.enable_deep_supervision = enable_deep_supervision        

        encoder_cfg, decoder_cfg = self._extract_encoder_decoder_cfg(string_tree)

        self.encoder = self._build_encoder(**encoder_cfg)
        self.decoder = self._build_decoder(**decoder_cfg)

    @staticmethod
    def get_norm_op(norm: str) -> Type[nn.Module]:
        # Normalization
        if norm == "instance_norm":
            return get_matching_instancenorm(nn.Conv3d)
        elif norm == "batch_norm":
            return get_matching_batchnorm(nn.Conv3d)
        else:
            raise ValueError(f"Normalization {norm} not supported.")

    @staticmethod
    def get_nonlin_op(nonlin: str)-> tuple[str, dict | None]:
        # Activation function
        if nonlin == "relu":
            return "torch.nn.ReLU", {"inplace": True}
        elif nonlin == "leaky_relu":
            return "torch.nn.LeakyReLU", {"inplace": True}
        elif nonlin == "elu":
            return "torch.nn.ELU", {"inplace": True}
        elif nonlin == "prelu":
            return "torch.nn.PReLU", None
        elif nonlin == "gelu":
            return "torch.nn.GELU", None
        else:
            raise ValueError(f"Activation function {nonlin} not supported.")

    @staticmethod
    def get_dropout_op(dropout: str) -> tuple[str | None, dict | None]:
        # Dropout
        if dropout == "dropout":
            dimension = convert_conv_op_to_dim(nn.Conv3d)
            dropout_op = get_matching_dropout(dimension=dimension)
            return dropout_op.__module__ + "." + dropout_op.__name__, {"p": self.hp_config.dropout_rate}
        elif dropout == "no_dropout":
            return None, None
        else:
            raise ValueError(f"Dropout {dropout} not supported.")

    @staticmethod
    def _extract_encoder_decoder_cfg(string_tree: str):
        def args_to_kwargs(args: list[str]) -> dict:
            kwargs = {}
            kwargs["norm"] = args[0]
            kwargs["nonlin"] = args[1]
            kwargs["dropout"] = args[2]
            kwargs["n_blocks_per_stage"] = CFGUNet.count_blocks_per_stage(args[3:])

            return kwargs
        
        pattern = r"(conv_encoder|res_encoder)\((.*?)\)|(conv_decoder|res_decoder)\((.*?)\)"
    
        matches = re.finditer(pattern, string_tree)
        for match in matches:
            if match.group(1): 
                encoder_cfg = {
                    "network_type": match.group(1),
                    **args_to_kwargs([item.strip() for item in match.group(2).split(",")])
                }
            elif match.group(3): 
                decoder_cfg = {
                    "network_type": match.group(3),
                    **args_to_kwargs([item.strip() for item in match.group(4).split(",")])
                }

        assert encoder_cfg is not None and decoder_cfg is not None, "Could not extract encoder and decoder configuration"
        
        return encoder_cfg, decoder_cfg

    @staticmethod
    def count_blocks_per_stage(blocks: list[str]) -> list[int]:
        counts = []
        current_count = 0

        for block in blocks:
            if block == "down" or block == "up":
                if current_count > 0:
                    counts.append(current_count)
                current_count = 0
            else:
                # We assume format "bX" where X is a number
                current_count += int(block[1:]) 
        
        # We also need to add the count for the last stage
        if current_count > 0:
            counts.append(current_count)
        
        return counts
    
    def _features_per_stage(self, num_stages) -> Tuple[int, ...]:
        return tuple([min(self.hp_config.max_features, self.hp_config.base_num_features * 2 ** i) for i in range(num_stages)])

    def _build_encoder(self, network_type: str, nonlin: str, norm: str, dropout: str, n_blocks_per_stage: list[int]) -> nn.Module:
        n_stages = len(n_blocks_per_stage)
        features_per_stage = self._features_per_stage(n_stages)

        norm_op = self.get_norm_op(norm)
        logger.info(f"Using encoder normalization: {norm_op}")

        nonlin_op, nonlin_op_kwargs = self.get_nonlin_op(nonlin)
        nonlin_op = pydoc.locate(nonlin_op)
        logger.info(f"Using encoder activation function: {nonlin_op}")

        dropout_op, dropout_op_kwargs = self.get_dropout_op(dropout)
        dropout_op = pydoc.locate(dropout_op) if dropout_op is not None else None
        logger.info(f"Using encoder dropout: {dropout_op}")

        logger.info(f"Using encoder n_blocks_per_stage: {n_blocks_per_stage}")

        if network_type == "conv_encoder":
            return PlainConvEncoder(
                input_channels=self.num_input_channels,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=self.architecture_kwargs["conv_op"],
                kernel_sizes=self.architecture_kwargs["kernel_sizes"][:n_stages],
                strides=self.architecture_kwargs["strides"][:n_stages],
                n_conv_per_stage=n_blocks_per_stage,
                conv_bias=self.architecture_kwargs["conv_bias"],
                norm_op=norm_op,
                norm_op_kwargs=self.architecture_kwargs["norm_op_kwargs"],
                dropout_op=dropout_op,                  # type: ignore
                dropout_op_kwargs=dropout_op_kwargs,    # type: ignore
                nonlin=nonlin_op,                       # type: ignore
                nonlin_kwargs=nonlin_op_kwargs,         # type: ignore
                return_skips=True
            )
        elif network_type == "res_encoder":
            return ResidualEncoder(
                input_channels=self.num_input_channels,
                n_stages=n_stages,
                features_per_stage=features_per_stage,
                conv_op=self.architecture_kwargs["conv_op"],
                kernel_sizes=self.architecture_kwargs["kernel_sizes"][:n_stages],
                strides=self.architecture_kwargs["strides"][:n_stages],
                n_blocks_per_stage=n_blocks_per_stage,
                conv_bias=self.architecture_kwargs["conv_bias"],
                norm_op=norm_op,
                norm_op_kwargs=self.architecture_kwargs["norm_op_kwargs"],
                dropout_op=dropout_op,                  # type: ignore
                dropout_op_kwargs=dropout_op_kwargs,    # type: ignore
                nonlin=nonlin_op,                       # type: ignore
                nonlin_kwargs=nonlin_op_kwargs,         # type: ignore
                return_skips=True
            )
        else:
            raise ValueError(f"Network type {network_type} not supported.")

    def _build_decoder(self, network_type: str, nonlin: str, norm: str, dropout: str, n_blocks_per_stage: list[int]) -> nn.Module:
        norm_op = self.get_norm_op(norm)
        logger.info(f"Using decoder normalization: {norm_op}")

        nonlin_op, nonlin_op_kwargs = self.get_nonlin_op(nonlin)
        nonlin_op = pydoc.locate(nonlin_op)
        logger.info(f"Using decoder activation function: {nonlin_op}")

        dropout_op, dropout_op_kwargs = self.get_dropout_op(dropout)
        dropout_op = pydoc.locate(dropout_op) if dropout_op is not None else None
        logger.info(f"Using decoder dropout: {dropout_op}")

        logger.info(f"Using decoder n_blocks_per_stage: {n_blocks_per_stage}")

        if network_type == "conv_decoder":
            return UNetDecoder(
                encoder=self.encoder,                   # type: ignore
                num_classes=self.num_output_channels,
                n_conv_per_stage=n_blocks_per_stage,
                deep_supervision=self.enable_deep_supervision,
                norm_op=norm_op,
                norm_op_kwargs=self.architecture_kwargs["norm_op_kwargs"],
                dropout_op=dropout_op,                  # type: ignore
                dropout_op_kwargs=dropout_op_kwargs,    # type: ignore
                nonlin=nonlin_op,                       # type: ignore
                nonlin_kwargs=nonlin_op_kwargs,         # type: ignore
                conv_bias=self.architecture_kwargs["conv_bias"],
            )
        elif network_type == "res_decoder":
            return UNetResDecoder(
                encoder=self.encoder,                   # type: ignore                   
                num_classes=self.num_output_channels,
                n_conv_per_stage=n_blocks_per_stage,
                deep_supervision=self.enable_deep_supervision,
                norm_op=norm_op,
                norm_op_kwargs=self.architecture_kwargs["norm_op_kwargs"],
                dropout_op=dropout_op,                  # type: ignore
                dropout_op_kwargs=dropout_op_kwargs,    # type: ignore
                nonlin=nonlin_op,                       # type: ignore
                nonlin_kwargs=nonlin_op_kwargs,         # type: ignore
                conv_bias=self.architecture_kwargs["conv_bias"],
            )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def initialize(self, module):
        InitWeights_He(1e-2)(module)

        if "res" in self.string_tree:
            init_last_bn_before_add_to_0(module)