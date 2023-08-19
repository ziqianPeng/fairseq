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
from fairseq.models.transformer_factor_decoder import TransformerFactorDecoder


# @register_model and @register_model_architecture are mandatory to define custom modules
@register_model("transformer_factor")
class TransformerFactorModel(TransformerModel):
    """
    transformer model with attention factor
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        print('TEST:', self.args.factor_mode)

    @staticmethod
    def add_args(parser):
        # add model specific argument
        # fmt: off
        TransformerModel.add_args(parser)
        # parser.add_argument('--source-mask', type=str, default='past',
        #                     help="choose between past, future and all to mask source context")
        parser.add_argument('--factor-mode', type=str, default='past',
                        help="choose between past, future and all to apply attention factor to source context")
        parser.add_argument('--factor-base', type=float, default=0.9,
                        help="a float between 0 and 1")
        # fmt: on

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerFactorDecoder(
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

@register_model_architecture("transformer_factor", "transformer_factor")
def transformer_factor(args):
    # args.source_mask = getattr(args, "source_mask", 'past')
    args.factor_mode = getattr(args, "factor_mode", 'past')

    print('TEST (transformer_factor):',args.factor_mode)
    base_architecture(args)

