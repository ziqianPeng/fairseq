# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor

# adapted from LearnedPositionalEmbedding
import logging
logger = logging.getLogger(__name__)

class LearnedPositionalEmbeddingUnif(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """
    # ziqian add position idx offset, positions begin at offset + padding_idx, 2024-03-27
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset: int = 0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1 - offset
        else:
            self.max_positions = self.num_embeddings - offset
        self.offset = offset
        logger.info(f"embed_position.max_positions() = {self.max_positions}")

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1) + self.offset ))
                # logger.info(f"ZP incrementai_states: offset = {self.offset}, positions = {positions}")
            else:
                positions = utils.make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace, offset= self.offset
                )
                # logger.info(f"ZP offset = {self.offset}, positions = {positions[:,-2:]}")
                # logger.info(f"ZP offset = {self.offset}, positions = {positions[:,:2]}")

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
