# adapted from 
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
)
from fairseq.models.transformer_mask_decoder import TransformerMaskDecoder

# TODO
# @register_model and @register_model_architecture are mandatory to define custom modules
@register_model("transformer_mask")
class TransformerMaskModel(TransformerModel):
    """
    transformer model with source mask
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        print('TEST:', self.args.source_mask)

    @staticmethod
    def add_args(parser):
        # add model specific argument
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--source-mask', type=str, default='past',
                            help="choose between past and future to mask past or future context")
        # fmt: on

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerMaskDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )


    def forward(self, src_tokens, src_lengths, sep_idx_src, sep_idx_tgt, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.decoder(
            prev_output_tokens, encoder_out,  sep_idx_src = sep_idx_src, sep_idx_tgt = sep_idx_tgt
            ) 



@register_model_architecture("transformer_mask", "transformer_mask")
def transformer_mask(args):
    args.source_mask = getattr(args, "source_mask", None) # TODO ziqian change None to past/future
    print('TEST (transformer_mask):',args.source_mask)
    base_architecture(args)

# @register_model_architecture("transformer_mask", "transformer_mask_past")
# def transformer_mask_past(args):
#     args.source_mask = getattr(args, "source_mask", 'past') # TODO ziqian change None to past/future
#     base_architecture(args)

