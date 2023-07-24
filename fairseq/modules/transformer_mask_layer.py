# adapted from fairseq/examples/attention_head_selection
# attention matrix = softmax(QK + context_mask), where context_mask is filled with -inf

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.utils import safe_getattr
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
import torch
from torch import Tensor
from typing import List, Optional, Dict 

from fairseq.modules.multihead_attention_3d import MultiheadAttention3DMask


class BatchMask(object):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    """
    def __init__(
        self, 
        sep_idx_src, 
        sep_idx_tgt,
        src_len,
        tgt_len,
        pad_idx_sep = -2,
        pad_left_src: Tensor = None,
        pad_left_tgt: Tensor = None,
        num_heads : int = 8,
        mask_mode : str = 'past',
        has_incremental_state : bool = False, 
        ):
        # assert mask_mode in ['past', 'future' 'mix'], mask_mode
        self.sep_idx_src = sep_idx_src
        self.sep_idx_tgt = sep_idx_tgt
        assert len(self.sep_idx_src) == len(self.sep_idx_tgt), f"for src: {sep_idx_src}, for tgt: {sep_idx_tgt}"
        # attention matrix (segment) of shape T_tgt x T_src 
        self.src_len = src_len
        self.tgt_len = tgt_len
        # for padding of sep indices
        self.pad_idx_sep = pad_idx_sep
        self.pad_left_src = pad_left_src
        self.pad_left_tgt = pad_left_tgt
        # attention matrix (batch) of shape N*nhead x T_tgt x T_src 
        self.num_heads = num_heads
        self.mask_mode = mask_mode
        self.has_incremental_state = has_incremental_state
    
    def _make_source_mask_seg_incremental(self, idx):
        # for a single document
        mask = torch.zeros(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]

        # in case of incremental state
        assert self.tgt_len == 1    
        if self.mask_mode == 'past':
            # +1 to mask <sep>, mask nothing if sep_tgt = [0]
            if len(sep_tgt) > 1:
                mask_idx = leftpad_src + sep_src[min( len(sep_tgt), len(sep_src) )-1] 

                # mask[:, : mask_idx + 1] = torch.tensor(True) 
                mask[:, : mask_idx ] = torch.tensor(True) 
        elif self.mask_mode == 'future':
            if len(sep_tgt) < len(sep_src):
                # mask_idx = leftpad_src + sep_src[min( len(sep_tgt), len(sep_src) )-1] 
                # if don't mask current src <sep>
                mask_idx = leftpad_src + sep_src[len(sep_tgt)-1] +1 

                mask[ :, mask_idx : ] = torch.tensor(True)
        else:
            raise NotImplementedError

        return mask.repeat( self.num_heads, 1, 1)

    
    def _make_source_mask_seg(self, idx):
        # for a single document
        mask = torch.zeros(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        for i in range(len(sep_tgt)-1 ):
            if self.mask_mode == 'past':
                begin_tgt = leftpad_tgt + sep_tgt[i+1] 
                if i+1 >= len(sep_src):
                    begin_src = leftpad_src + sep_src[0]
                    mask[begin_tgt:, begin_src : ] = torch.tensor(True)
                else:
                    begin_src = leftpad_src + sep_src[i] 

                    # end_src = leftpad_src + sep_src[i+1] +1 # +1 to mask also <sep> in previous src sent
                    end_src = leftpad_src + sep_src[i+1] 
                    mask[begin_tgt:, begin_src : end_src] = torch.tensor(True)

            elif self.mask_mode == 'future':
                # if i+1 >= len(sep_src), no future token to mask, all tokens are at past
                if i+1 < len(sep_src):
                    begin_tgt = leftpad_tgt + sep_tgt[i]
                    end_tgt = leftpad_tgt + sep_tgt[i+1] +1 # +1 to update mask also for tgt <sep>
                    
                    # begin_src = leftpad_src + sep_src[i+1]  # if mask <sep> of current src sentence:
                    begin_src = leftpad_src + sep_src[i+1] +1 

                    mask[begin_tgt: end_tgt, begin_src : ] = torch.tensor(True)

            else:
                raise NotImplementedError

        # if self.src_len < 50 and self.sep_idx_tgt.size(1) > 1 and idx < 4:
        #     print(f'TEST idx {idx}, src_len { self.src_len} tgt_len {self.tgt_len}')
        #     print(f'TEST src_leftpad { leftpad_src} tgt_leftpad {leftpad_tgt}')
        #     print('TEST sep_idx_src', self.sep_idx_src)
        #     print('TEST sep_idx_tgt', self.sep_idx_tgt)
        #     print('TEST context mask')
        #     for i in range(len(mask)):
        #         print(i, mask[i])

        return mask.repeat( self.num_heads, 1, 1)
        # TODO expand or repeat??  # expand may better for memory complexity, 
        # but fairseq used `repeat` in MultiHeadAttn.forward for attn_mask
        # if using tensor.expand, avoid direct in place operation with it (make a copy)
        # return mask.unsqueeze(0).expand( self.num_heads, -1, -1)
        
    def make_source_mask(self):
        # for a batch, use map
        if self.has_incremental_state:
            batch_mask = list(map(self._make_source_mask_seg_incremental, range(len(self.sep_idx_tgt))))
        else:
            batch_mask = list(map(self._make_source_mask_seg, range(len(self.sep_idx_tgt))))

        return torch.cat(batch_mask).to(self.sep_idx_tgt.device)


class MaskedTransformerDecoderLayer(TransformerDecoderLayer):
    """
    if need to change MHA type, maybe add an argument attn_type in ['enc_mask', 'mask_enc']
    then choose MHA class in build_encoder_attention wrt attn_type
    """

    def __init__(
        self,
        args,
        layer_idx,
        self_attn_head_selector=None,
        enc_attn_head_selector=None,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
    

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention3DMask(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )


    # copied from transformer_layer.forward, pass <sep> indice and pad_left_src/tgt by argument
    # make future/past mask and pass it to self.encoder_attn
    def forward(
        self,
        x,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        sep_idx_src: Optional[Tensor] = None,
        sep_idx_tgt: Optional[Tensor] = None,
        mask_mode: str = 'past',
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            ##### make batch past / future mask
            # 1. number of padding at left:
            pad_left_src: Tensor= None
            pad_left_tgt: Tensor = None
            # B X T
            if encoder_padding_mask is not None and encoder_padding_mask[:, 0].any():
                pad_left_src = encoder_padding_mask.sum(axis = 1)

            if self_attn_padding_mask is not None and self_attn_padding_mask[:, 0].any():
                pad_left_tgt = self_attn_padding_mask.sum(axis = 1)
                
            # 2. make mask
            # should be N*num_head, T_tgt, T_src
            context_mask: Tensor = None
            if sep_idx_src.size(1) > 1:
                # if there is more than 1 sentence in the document 
                context_mask = BatchMask(
                    sep_idx_src, 
                    sep_idx_tgt,
                    src_len = encoder_padding_mask.size(1),
                    tgt_len = x.size(0),
                    pad_idx_sep = -2,
                    pad_left_src = pad_left_src ,
                    pad_left_tgt = pad_left_tgt,
                    num_heads = self.nh,
                    mask_mode = mask_mode,
                    has_incremental_state = incremental_state is not None,
                    ).make_source_mask().to(encoder_padding_mask.device)
            
            ##### pass mask to encoder_attn 
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_mask = context_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None
        


    
