# adapted from transformer_mask_decoder.py

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Any
from torch import Tensor

from fairseq.distributed import fsdp_wrap
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_factor_layer import FadedTransformerDecoderLayer
from fairseq.modules.batch_attention_factor import (
    BatchAttnFactorSentAll,
    BatchAttnFactorSentFuture,
    BatchAttnFactorSentPast,
)


class TransformerFactorDecoder(TransformerDecoder):
    """
    the main modifiations (from TransformerDecoder) are:
    - build_decoder_layer: to apply TransformerMaskDecoderLayerBase
    - copy then modified forward / extract_feature / extract_feature_scriptable only to pass 
      the indice of <sep> tag to decoder layer for the masked cross attention
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):

        super().__init__(
            args, dictionary, embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection
        )
        print('TEST padding_idx decoder: ', self.padding_idx)
        self.factor_mode = args.factor_mode
        self.factor_base = args.factor_base
        assert self.factor_base > 0 and self.factor_base < 1, "invalid value for attention factor"


    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        # replace TransformerDecoderLayerBase by FadedTransformerDecoderLayer
        layer = FadedTransformerDecoderLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
        
    def set_batch_factor(self):
        if self.factor_mode == 'future':
            return BatchAttnFactorSentFuture
        elif self.factor_mode == 'past':
            return BatchAttnFactorSentPast
        elif self.factor_mode == 'all':
            return BatchAttnFactorSentAll
        else:
            raise NotImplementedError

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        sep_idx_src: Optional[Tensor] =None,
        sep_idx_tgt: Optional[Tensor] = None,
        update_context: Optional[bool] = True,
    ):
    # the same as TransformerMaskDecoder(Base)
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            update_context = update_context,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        sep_idx_src: Optional[Tensor] =None,
        sep_idx_tgt: Optional[Tensor] = None,
        update_context : Optional[bool] = True,
    ):
    # the same as TransformerDecoder(Base), rewrite to pass <sep> indices to decoder layer 
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            update_context = update_context,
        )

    # also copy extract_features_scriptable, but only modifie the layer(...) part
    # to pass <sep> indices to decoder layer for the masked cross-attention
    # afterward, in layer, use padding mask & self_attn_padding_mask to determine whether source/target is padded at left
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        sep_idx_src: Optional[Tensor] =None,
        sep_idx_tgt: Optional[Tensor] = None,
        update_context : Optional[bool] = True,
    ):
        """
        Similar to *forward* but only return features. read more information in TransformerDecoder

        Args:
            sep_idx_src (LongTensor): the indice of sep in batch (source)
            sep_idx_tgt (LongTensor): the indice of sep in batch (target)

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None

        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0] # B x T

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
        
        if incremental_state is not None:
            # take the last column of prev_output_tokens
            prev_output_tokens = prev_output_tokens[:, -1:] 
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        ##### make batch past / future mask                
        context_factor: Tensor = None
        if sep_idx_src.size(1) > 1:
        # if there is a document with more than 1 sentence in the  batch
            if update_context:
                # 1. number of padding at left:
                pad_left_src: Tensor= None
                pad_left_tgt: Tensor = None
                # B X T
                if padding_mask is not None and padding_mask[:, 0].any():
                    pad_left_src = padding_mask.sum(axis = 1)

                if self_attn_padding_mask is not None and self_attn_padding_mask[:, 0].any():
                    pad_left_tgt = self_attn_padding_mask.sum(axis = 1)

                # 2. make factor matrice
                # should be N*num_head, T_tgt, T_src
                batch_factor = self.set_batch_factor()
                context_factor = batch_factor(
                    sep_idx_src, 
                    sep_idx_tgt,
                    src_len = padding_mask.size(1),
                    tgt_len = x.size(0),
                    pad_idx_sep = -2,
                    pad_left_src = pad_left_src ,
                    pad_left_tgt = pad_left_tgt,
                    num_heads = self.cfg.decoder.attention_heads,
                    factor_base = self.factor_base,
                    has_incremental_state = incremental_state is not None,
                    ).get_attn_factor().to(padding_mask.device, dtype = x.dtype)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # mask should be the same for all layer, no need to update it for idx > 0
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                context_factor = context_factor,
                save_context = sep_idx_src.size(1) > 1 and update_context,
                incremental_context_factor = sep_idx_src.size(1) > 1 and not update_context,
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


