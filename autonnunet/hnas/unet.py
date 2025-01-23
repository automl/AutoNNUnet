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
    convert_conv_op_to_dim, get_matching_batchnorm,
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

        parsed_tree = self.parse_nested_brackets(string_tree)
        encoder_cfg, decoder_cfg = self.extract_architecture_cfg(parsed_tree)

        self.encoder = self._build_encoder(**encoder_cfg)
        self.decoder = self._build_decoder(**decoder_cfg)

    @staticmethod
    def parse_nested_brackets(string_tree: str) -> list:
        def helper(tokens):
            result = []
            while tokens:
                token = tokens.pop(0)
                if token == '(':
                    # We need to start a new nested list
                    result.append(helper(tokens))
                elif token == ')':
                    # We end the current nested list
                    return result
                else:
                    # We add the token to the current list
                    result.append(token)
            return result

        # Tokenize the string
        tokens = re.findall(r'\(|\)|[^()\s]+', string_tree)
        return helper(tokens)
    
    @staticmethod
    def parse_encoder_decoder(sublist: list) -> dict:
        result = {
            "network_type": sublist[1],
            "norm": sublist[2][1],
            "nonlin": sublist[3][1],
            "dropout": sublist[4][1],
        }
        
        blocks_per_stage = []
        blocks_list = sublist[5:]
        for b in blocks_list:
            if isinstance(b, str):
                continue
            else:
                n_blocks = int(b[1].replace("b", ""))
                blocks_per_stage.append(n_blocks)
        result["n_blocks_per_stage"] = blocks_per_stage

        return result

    @staticmethod
    def extract_architecture_cfg(nested_list: list) -> tuple[dict, dict]:
        # First, we remove the unused parts of the nested list
        while len(nested_list) == 1:
            nested_list = nested_list[0]

        # Now we have the initial structure and the encoder and decoder lists
        assert len(nested_list) == 4

        encoder_cfg = CFGUNet.parse_encoder_decoder(nested_list[2])
        decoder_cfg = CFGUNet.parse_encoder_decoder(nested_list[3])

        return encoder_cfg, decoder_cfg
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
            return "torch.nn.PReLU", {}
        elif nonlin == "gelu":
            return "torch.nn.GELU", {}
        else:
            raise ValueError(f"Activation function {nonlin} not supported.")

    def get_dropout_op(self, dropout: str) -> tuple[str | None, dict | None]:
        # Dropout
        if dropout == "dropout":
            dimension = convert_conv_op_to_dim(nn.Conv3d)
            dropout_op = get_matching_dropout(dimension=dimension)
            return dropout_op.__module__ + "." + dropout_op.__name__, {"p": self.hp_config.dropout_rate}
        elif dropout == "no_dropout":
            return None, {}
        else:
            raise ValueError(f"Dropout {dropout} not supported.")
        
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

    def _build_decoder(self, network_type: str, nonlin: str, norm: str, dropout: str, n_blocks_per_stage: list[int]) -> UNetDecoder | UNetResDecoder:
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
        else:
            raise ValueError(f"Network type {network_type} not supported.")

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def initialize(self, module):
        InitWeights_He(1e-2)(module)

        if "res" in self.string_tree:
            init_last_bn_before_add_to_0(module)
