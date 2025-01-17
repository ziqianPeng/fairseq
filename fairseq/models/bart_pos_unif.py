"""
adapted from fairseq/model/bart/model.py
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.models.bart import (
    BARTModel,
    mbart_large_architecture,
    bart_large_architecture,
    )

from fairseq.modules.transformer_sentence_encoder import init_bert_params



logger = logging.getLogger(__name__)


@register_model("bart_pos_unif")
class BARTPosUnifModel(BARTModel):

    def __init__(self, args, encoder, decoder):
        # here args is TransformerConfig
        super().__init__(args, encoder, decoder)
        logger.info(f"position offset = {self.args.offset}")

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_offsets: Optional[torch.Tensor] = None,
        tgt_offsets: Optional[torch.Tensor] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            src_offsets = src_offsets,
        )

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            tgt_offsets = tgt_offsets,
        )
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[src_tokens.eq(eos), :].view(
                x.size(0), -1, x.size(-1)
            )[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break
        return x, extra

@register_model_architecture("bart_pos_unif", "bart_large_pos_unif")
def bart_large_pos_unif_architecture(args):
    args.active_uniform_pos = getattr(args, "active_uniform_pos", False)
    args.share_pos_offset_enc_dec = getattr(args, "share_pos_offset_enc_dec", False)
    bart_large_architecture(args)



@register_model_architecture("bart_pos_unif", "mbart_large_pos_unif")
def mbart_large_pos_unif_architecture(args):
    args.active_uniform_pos = getattr(args, "active_uniform_pos", False)
    args.share_pos_offset_enc_dec = getattr(args, "share_pos_offset_enc_dec", False)
    mbart_large_architecture(args)

