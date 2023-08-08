# TODO ziqian adapt batch_mask.py to make attention factor
# TODO ziqian register this in modules/__init__.py
# make attention weight factor with respect to the sentence boundary
# attention matrix = softmax(QK * attn_factor_matrix), with element-wise multiplication


from fairseq.utils import safe_getattr

import torch
from torch import Tensor
from typing import List, Optional, Dict 

class BatchAttnFactor(object):
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
        factor_base : float = 0.9,
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
    
    def _make_attn_factor_seg_incremental(self, idx):
        """in case of using incremental_state (self.tgt_len = 1)

        Args:
            idx (int): index of current document in batch
        """
        raise NotImplementedError

    def _make_attn_factor_seg(self, idx):
        """in case 

        Args:
            idx (int): index of current document in batch
        """
        raise NotImplementedError
    
    def get_attn_factor(self):
        # for a batch, use map
        if self.has_incremental_state:
            assert self.tgt_len == 1
            batch_mask = list(map(self._make_attn_factor_seg_incremental, range(len(self.sep_idx_tgt))))
        else:
            batch_mask = list(map(self._make_attn_factor_seg, range(len(self.sep_idx_tgt))))

        return torch.cat(batch_mask).to(self.sep_idx_tgt.device)
    
    def test_verbose(self, idx, mask, leftpad_src, leftpad_tgt = None):
        if self.src_len < 30 and self.sep_idx_tgt.size(1) > 1 and idx < 4:
            print(f'TEST idx {idx}, src_len { self.src_len} tgt_len {self.tgt_len}')
            print(f'TEST src_leftpad { leftpad_src}')
            if leftpad_tgt:
                print(f'TEST tgt_leftpad {leftpad_tgt}')
            print('TEST sep_idx_src', self.sep_idx_src)
            print('TEST sep_idx_tgt', self.sep_idx_tgt)
            print('TEST context mask')
            for i in range(len(mask)):
                print(i, mask[i])



# class BatchAttnFactorSent(BatchAttnFactor):
#     """
#     size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
#     
#       factor_type:
#           1. 10**(-d)
#           2. (90%)*(d)
#     """
#     def __init__(
#         self, 
#         sep_idx_src, 
#         sep_idx_tgt,
#         src_len,
#         tgt_len,
#         pad_idx_sep = -2,
#         pad_left_src: Tensor = None,
#         pad_left_tgt: Tensor = None,
#         num_heads : int = 8,
#         has_incremental_state : bool = False, 
#         ):
#         super().__init__(
#             sep_idx_src = sep_idx_src,
#             sep_idx_tgt = sep_idx_tgt,
#             src_len = src_len,
#             tgt_len = tgt_len,
#             pad_idx_sep = pad_idx_sep,
#             pad_left_src = pad_left_src,
#             pad_left_tgt = pad_left_tgt,
#             num_heads = num_heads,
#             mask_mode = 'factor',
#             has_incremental_state = has_incremental_state,
#             )

    
    # def _make_attn_factor_seg_incremental(self, idx):

    # def _make_attn_factor_seg(self, idx):



# class BatchAttnFactorTok(BatchAttnFactor):
#     """
#     size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
#     """
#     def __init__(
#         self, 
#         sep_idx_src, 
#         sep_idx_tgt,
#         src_len,
#         tgt_len,
#         pad_idx_sep = -2,
#         pad_left_src: Tensor = None,
#         pad_left_tgt: Tensor = None,
#         num_heads : int = 8,
#         has_incremental_state : bool = False, 
#         ):
#         super().__init__(
#             sep_idx_src = sep_idx_src,
#             sep_idx_tgt = sep_idx_tgt,
#             src_len = src_len,
#             tgt_len = tgt_len,
#             pad_idx_sep = pad_idx_sep,
#             pad_left_src = pad_left_src,
#             pad_left_tgt = pad_left_tgt,
#             num_heads = num_heads,
#             mask_mode = 'factor',
#             has_incremental_state = has_incremental_state,
#             )

    
    # def _make_attn_factor_seg_incremental(self, idx):

    # def _make_attn_factor_seg(self, idx):
