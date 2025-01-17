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
import numpy as np

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
    # ziqian add pseudo uniform position using offset, positions begin at offset + padding_idx, 2024-03-27
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset:int = 0):
        # offset applied during inference
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1 - offset
        else:
            self.max_positions = self.num_embeddings - offset
        self.offset = offset
        # todo remove offset here
        logger.info(f"posUnif: embed_position.max_positions() = {self.max_positions}")

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
        deactive_pos_unif: Optional[bool] = None,
        input_offsets: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = self.make_positions_uniform_incremental(
                    input, deactive_pos_unif = deactive_pos_unif, input_offsets = input_offsets 
                    )
                # logger.info(f"ZP incrementai_states: offset = {self.offset}, positions = {positions}, size = {positions.size()}")
                # logger.info(f"ZP uniform pos input_offsets.size() = {input_offsets.size()}, input_offsets = {input_offsets}")
                # logger.info(f"ZP incrementai_states:input.size() = {input.size()}")
                # logger.info(f"input {input}")

            else:
                positions = self.make_positions_uniform(
                    input, self.padding_idx, deactive_pos_unif, input_offsets,
                )
                # logger.info(f"input {input}")
                # logger.info(f"ZP uniform pos input_offsets = {input_offsets}")
                # logger.info(f"ZP uniform pos deactive_pos_unif = {deactive_pos_unif}, positions[:,:3] = {positions[:,:3]}")

        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
    
    # ziqian test uniform position using offset 2024-05-15
    def make_positions_uniform(self, tensor, padding_idx: int, deactive_pos_unif: bool = None, input_offsets = None,):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        offset : position index of input sequence begins with offset + padding_idx for each input
        offset can be a integer of a list of integer
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()

        active_ratio = np.random.rand(1)[0] 
        threshold = 0.3 #TODO add a decay
        if deactive_pos_unif or input_offsets is None or active_ratio < 0.3:        
            # logger.info(f'DEBUG test deactive uniform position offsets during inference, deactive_pos_unif = {deactive_pos_unif} ')
            # return ( (torch.cumsum(mask, dim=1).type_as(mask) + self.offset) * mask).long() + padding_idx 
            return ( (torch.cumsum(mask, dim=1).type_as(mask) ) * mask).long() + padding_idx 

        cumsum_mask = torch.cumsum(mask, dim=1).type_as(mask)
        batch_offsets = input_offsets.view(len(input_offsets), 1).detach().clone()
        batch_offsets = batch_offsets.to(device = tensor.device)
        # logger.info(f"ZP uniform pos batch_offsets = {batch_offsets}")
        res = ( ( cumsum_mask + batch_offsets ) * mask).long() + padding_idx 
        del batch_offsets
        return res
    
    def make_positions_uniform_incremental(self, input, deactive_pos_unif = None, input_offsets = None, ):
        positions = torch.zeros(
            (1, 1), device=input.device, dtype=input.dtype
        ).fill_(int(self.padding_idx + input.size(1) + self.offset ))

        if not deactive_pos_unif:
            assert(input_offsets is not None)
            # assert(input_offsets is not None and input_offsets.size() == positions )
            # beam_size = input.size()[0] // len(input_offsets)
            # # incremental_offsets = input_offsets.view(-1, 1).repeat(1, beam_size).view(-1,1).clone()
            # incremental_offsets = incremental_offsets.to(device = input.device)
            incremental_offsets = input_offsets.clone().to(device = input.device)
            positions = positions + incremental_offsets
            # logger.info(f"DEBUG incremental_offsets= {incremental_offsets}")
            del incremental_offsets
            # logger.info(f"DEBUG positions= {positions}")

        return positions
    

