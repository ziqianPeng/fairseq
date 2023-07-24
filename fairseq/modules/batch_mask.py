# make context mask with respect to the sentence boundary
# attention matrix = softmax(QK + context_mask), where context_mask is filled with -inf


from fairseq.utils import safe_getattr

import torch
from torch import Tensor
from typing import List, Optional, Dict 




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
            if len(sep_tgt) > len(sep_src):
                # only </s> is not masked if more target segments are generated
                mask[:, leftpad_src : -1 ] = torch.tensor(True)
                return mask.repeat( self.num_heads, 1, 1)
            
            if len(sep_tgt) > 1:
                # +1 to mask <sep>, mask nothing if sep_tgt = [0] (no past)
                mask_idx = leftpad_src + sep_src[len(sep_tgt)-1] 
                mask[:, : mask_idx + 1] = torch.tensor(True) 

        elif self.mask_mode == 'future':
            # if len(sep_tgt) == len(sep_src), then no future for the current sentence
            if len(sep_tgt) < len(sep_src):
                mask_idx = leftpad_src + sep_src[ len(sep_tgt)-1] 
                # if don't mask current src <sep>
                # mask_idx = leftpad_src + sep_src[len(sep_tgt)-1] +1 

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
                    # only </s> is not masked 
                    mask[begin_tgt:, leftpad_src : -1 ] = torch.tensor(True)
                else:
                    begin_src = leftpad_src + sep_src[i] 
                    end_src = leftpad_src + sep_src[i+1] +1 # +1 to mask also <sep> in previous src sent
                    # end_src = leftpad_src + sep_src[i+1] 
                    mask[begin_tgt:, begin_src : end_src] = torch.tensor(True)

            elif self.mask_mode == 'future':
                # if i+1 >= len(sep_src), no future token to mask, all tokens are at past
                # TODO ziqian check the case that len(sep_tgt) = 1, but len(sep_src) > 1
                if i+1 < len(sep_src):
                    begin_tgt = leftpad_tgt + sep_tgt[i]
                    end_tgt = leftpad_tgt + sep_tgt[i+1] +1 # +1 to update mask also for tgt <sep>
                    
                    begin_src = leftpad_src + sep_src[i+1]  # if mask <sep> of current src sentence:
                    # begin_src = leftpad_src + sep_src[i+1] +1 

                    mask[begin_tgt: end_tgt, begin_src : ] = torch.tensor(True)

            else:
                raise NotImplementedError

        return mask.repeat( self.num_heads, 1, 1)

    def make_source_mask(self):
        # for a batch, use map
        if self.has_incremental_state:
            batch_mask = list(map(self._make_source_mask_seg_incremental, range(len(self.sep_idx_tgt))))
        else:
            batch_mask = list(map(self._make_source_mask_seg, range(len(self.sep_idx_tgt))))

        return torch.cat(batch_mask).to(self.sep_idx_tgt.device)



# TODO ziqian test it
class BatchMaskAllSource(BatchMask):
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
        # assume sep_idx_src.size(1) > 1 
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
        mask = torch.ones(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]

        # in case of incremental state
        assert self.tgt_len == 1    
        # +1 to mask <sep>, mask nothing if sep_tgt = [0]
        len_tgt = len(sep_tgt)
        if len_tgt == 1:
            #only have future
            mask[:, sep_src[1]] = torch.tensor(False)
            return mask.repeat( self.num_heads, 1, 1)
        
        if len_tgt  < len(sep_src):
            mask[:, leftpad_src + sep_src[len_tgt -2]+1 : leftpad_src + sep_src[len_tgt-1] ] = torch.tensor(False)
                
        if len_tgt == len(sep_src):
            # only have past, no future
            mask[:, leftpad_src + sep_src[len_tgt-1] + 1 : ] = torch.tensor(False)

        return mask.repeat( self.num_heads, 1, 1)

    def _make_source_mask_seg(self, idx):
        # for a single document
        mask = torch.ones(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        for i in range(len(sep_tgt)-1 ):
            begin_tgt = leftpad_tgt + sep_tgt[i] +1 if i > 0  else leftpad_tgt 
            end_tgt = leftpad_tgt + sep_tgt[i+1] 

            begin_src = leftpad_src + sep_src[i] +1 if i > 0  else leftpad_src
            end_src = leftpad_src + sep_src[i+1] 

            mask[begin_tgt : end_tgt, begin_src : end_src] = torch.tensor(False)

        # last sentence
        mask[leftpad_tgt + sep_tgt[-1]+1 : ,  leftpad_src + sep_src[-1] +1 : ] = torch.tensor(False)

        # TODO check the case that len(sep_src) > 1 but len(sep_tgt) = 1, and the case len(sep_tgt) > len(sep_src)
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

        return mask.repeat( self.num_heads, 1, 1)


   