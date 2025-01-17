# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from .learned_positional_embedding import LearnedPositionalEmbedding
from .learned_positional_embedding_offset import LearnedPositionalEmbeddingOffset
from .learned_positional_embedding_uniform import LearnedPositionalEmbeddingUnif

from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
    offset: int = 0,
    active_uniform_pos: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        # m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        if active_uniform_pos:
            m = LearnedPositionalEmbeddingUnif(num_embeddings, embedding_dim, padding_idx, offset)
        else:
            m = LearnedPositionalEmbeddingOffset(num_embeddings, embedding_dim, padding_idx, offset)

        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m
