"""The Context-Free Grammar U-Net."""
from __future__ import annotations

import logging
import pydoc
import re
from typing import TYPE_CHECKING

import torch
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_batchnorm,
    get_matching_dropout,
    get_matching_instancenorm,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import (
    UNetResDecoder,
)
from dynamic_network_architectures.initialization.weight_init import (
    InitWeights_He,
    init_last_bn_before_add_to_0,
)
from torch import nn

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger("CFGUNet")

class CFGUNet(nn.Module):
    """The U-Net based on a context-free grammar."""
    def __init__(
            self,
            hp_config: DictConfig,
            string_tree: str,
            arch_init_kwargs: dict,
            arch_kwargs_req_import: list[str],
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool       # noqa: FBT001
        ):
        """Initializes the CFGUNet.

        Parameters
        ----------
        hp_config : DictConfig
            The hyperparameters configuration.

        string_tree : str
            The string tree.

        arch_init_kwargs : dict
            The architecture initialization keyword arguments.

        arch_kwargs_req_import : list[str]
            The architecture keyword arguments requiring import.

        num_input_channels : int
            The number of input channels.

        num_output_channels : int
            The number of output channels.

        enable_deep_supervision : bool
            Whether to enable deep supervision.
        """
        super().__init__()

        self.hp_config = hp_config
        self.string_tree = string_tree

        self.architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_kwargs_req_import:
            if self.architecture_kwargs[ri] is not None:
                self.architecture_kwargs[ri] = pydoc.locate(
                    self.architecture_kwargs[ri]
                )

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.enable_deep_supervision = enable_deep_supervision

        parsed_tree = self.parse_nested_brackets(string_tree)
        encoder_cfg, decoder_cfg = self.extract_architecture_cfg(parsed_tree)

        self.encoder = self._build_encoder(**encoder_cfg)
        self.decoder = self._build_decoder(**decoder_cfg)

    @staticmethod
    def parse_nested_brackets(string_tree: str) -> list:
        """Parses the nested brackets of the string tree.

        Parameters
        ----------
        string_tree : str
            The string tree to parse.

        Returns:
        -------
        list
            The parsed nested brackets.
        """
        def helper(tokens):
            result = []
            while tokens:
                token = tokens.pop(0)
                if token == "(":    # noqa: S105
                    # We need to start a new nested list
                    result.append(helper(tokens))
                elif token == ")":  # noqa: S105
                    # We end the current nested list
                    return result
                else:
                    # We add the token to the current list
                    result.append(token)
            return result

        # Tokenize the string
        tokens = re.findall(r"\(|\)|[^()\s]+", string_tree)
        return helper(tokens)

    @staticmethod
    def parse_encoder_decoder(sublist: list) -> dict:
        """Parses the encoder or decoder configuration.

        Parameters
        ----------
        sublist : list
            The sublist containing the encoder or decoder configuration.

        Returns:
        -------
        dict
            The encoder or decoder configuration.
        """
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
            n_blocks = int(b[1].replace("b", ""))
            blocks_per_stage.append(n_blocks)
        result["n_blocks_per_stage"] = blocks_per_stage

        return result

    @staticmethod
    def extract_architecture_cfg(nested_list: list) -> tuple[dict, dict]:
        """Extracts the architecture configuration from the nested list.

        Parameters
        ----------
        nested_list : list
            The nested list containing the architecture configuration.

        Returns:
        -------
        tuple[dict, dict]
            The encoder and decoder configuration.
        """
        # First, we remove the unused parts of the nested list
        while len(nested_list) == 1:
            nested_list = nested_list[0]

        # Now we have the initial structure and the encoder and decoder lists
        assert len(nested_list) == 4    # noqa: PLR2004

        encoder_cfg = CFGUNet.parse_encoder_decoder(nested_list[2])
        decoder_cfg = CFGUNet.parse_encoder_decoder(nested_list[3])

        return encoder_cfg, decoder_cfg
    
    @staticmethod
    def get_norm_op(norm: str) -> type[nn.Module]:
        """Returns the normalization operation.

        Parameters
        ----------
        norm : str
            The normalization to use.

        Returns:
        -------
        type[nn.Module]
            The normalization operation.
        """
        # Normalization
        if norm == "instance_norm":
            return get_matching_instancenorm(nn.Conv3d)
        if norm == "batch_norm":
            return get_matching_batchnorm(nn.Conv3d)
        raise ValueError(f"Normalization {norm} not supported.")

    @staticmethod
    def get_nonlin_op(nonlin: str)-> tuple[str, dict | None]:
        """Returns the activation function and its arguments.

        Parameters
        ----------
        nonlin : str
            The activation function to use.

        Returns:
        -------
        tuple[str, dict | None]
            The activation function and its arguments.
        """
        # Activation function
        if nonlin == "relu":
            return "torch.nn.ReLU", {"inplace": True}
        if nonlin == "leaky_relu":
            return "torch.nn.LeakyReLU", {"inplace": True}
        if nonlin == "elu":
            return "torch.nn.ELU", {"inplace": True}
        if nonlin == "prelu":
            return "torch.nn.PReLU", {}
        if nonlin == "gelu":
            return "torch.nn.GELU", {}
        raise ValueError(f"Activation function {nonlin} not supported.")

    def get_dropout_op(
            self,
            dropout: str
        ) -> tuple[str | None, dict | None]:
        """Returns the dropout operation and its arguments.

        Parameters
        ----------
        dropout : str
            The dropout operation to use.

        Returns:
        -------
        tuple[str | None, dict | None]
            The dropout operation and its arguments
        """
        # Dropout
        if dropout == "dropout":
            dimension = convert_conv_op_to_dim(nn.Conv3d)
            dropout_op = get_matching_dropout(dimension=dimension)
            return dropout_op.__module__ + "." +\
                   dropout_op.__name__, {"p": self.hp_config.dropout_rate}
        if dropout == "no_dropout":
            return None, {}
        raise ValueError(f"Dropout {dropout} not supported.")

    def _features_per_stage(self, num_stages) -> tuple[int, ...]:
        """Computes the number of features per stage.

        Parameters
        ----------
        num_stages : int
            The number of stages in the network.

        Returns:
        -------
        tuple[int, ...]
            The number of features per stage.
        """
        return tuple(
            [
                min(
                    self.hp_config.max_features,
                    self.hp_config.base_num_features * 2 ** i
                ) for i in range(num_stages)
            ]
        )

    def _build_encoder(
            self,
            network_type: str,
            nonlin: str,
            norm: str,
            dropout: str,
            n_blocks_per_stage: list[int]
        ) -> nn.Module:
        """Creates the encoder part of the network.

        Parameters
        ----------
        network_type : str
            The type of network to build.

        nonlin : str
            The activation function to use.

        norm : str
            The normalization to use.

        dropout : str
            The dropout to use.

        n_blocks_per_stage : list[int]
            The number of blocks per stage.

        Returns:
        -------
        nn.Module
            The encoder part of the network.
        """
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
        if network_type == "res_encoder":
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
        raise ValueError(f"Network type {network_type} not supported.")

    def _build_decoder(
            self,
            network_type: str,
            nonlin: str,
            norm: str,
            dropout: str,
            n_blocks_per_stage: list[int]
        ) -> UNetDecoder | UNetResDecoder:
        """Creates the decoder part of the network.

        Parameters
        ----------
        network_type : str
            The type of network to build.

        nonlin : str
            The activation function to use.

        norm : str
            The normalization to use.

        dropout : str
            The dropout to use.

        n_blocks_per_stage : list[int]
            The number of blocks per stage.

        Returns:
        -------
        UNetDecoder | UNetResDecoder
            The decoder part of the network.
        """
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
        if network_type == "res_decoder":
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
        raise ValueError(f"Network type {network_type} not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns:
        -------
        torch.Tensor
            The output tensor.
        """
        skips = self.encoder(x)
        return self.decoder(skips)

    def initialize(self, module: nn.Module) -> None:
        """Initializes the weights of the network.

        Parameters
        ----------
        module : nn.Module
            The module to initialize.
        """
        InitWeights_He(1e-2)(module)

        if "res" in self.string_tree:
            init_last_bn_before_add_to_0(module)
