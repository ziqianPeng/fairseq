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
        """in case of using incremental_state (self.tgt_len = 1)

        Args:
            idx (int): index of current document in batch
        """
        raise NotImplementedError

    def _make_source_mask_seg(self, idx):
        """in case 

        Args:
            idx (int): index of current document in batch
        """
        raise NotImplementedError
    
    def make_source_mask(self):
        # for a batch, use map
        if self.has_incremental_state:
            assert self.tgt_len == 1
            batch_mask = list(map(self._make_source_mask_seg_incremental, range(len(self.sep_idx_tgt))))
        else:
            batch_mask = list(map(self._make_source_mask_seg, range(len(self.sep_idx_tgt))))

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

# for "return mask.repeat( self.num_heads, 1, 1)    "            
#         # TODO expand or repeat??  # expand may better for memory complexity, 
#         # but fairseq used `repeat` in MultiHeadAttn.forward for attn_mask
#         # if using tensor.expand, avoid direct in place operation with it (make a copy)
#         # return mask.unsqueeze(0).expand( self.num_heads, -1, -1)


class BatchMaskPast(BatchMask):
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
        has_incremental_state : bool = False, 
        ):
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            mask_mode = 'past',
            has_incremental_state = has_incremental_state,
            )


    def _make_source_mask_seg_incremental(self, idx):
        # maskPast:  in case of incremental state,  self.tgt_len = 1
        mask = torch.zeros(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   
        
        if len(sep_tgt) > len(sep_src):
            # only </s> is not masked if more target segments are generated
            mask[:, leftpad_src : -1 ] = torch.tensor(True)

            # print('TEST sep_tgt>sep_src', idx)
            # self.test_verbose( idx, mask, leftpad_src)

            return mask.repeat( self.num_heads, 1, 1)
            
        if len(sep_tgt) > 1:
            # +1 to mask <sep>, mask nothing if sep_tgt = [0] (no past)
            mask_idx = leftpad_src + sep_src[len(sep_tgt)-1] 
            # mask[:, : mask_idx ] = torch.tensor(True)
            mask[:, leftpad_src : mask_idx + 1] = torch.tensor(True)

        # self.test_verbose(idx, mask, leftpad_src, leftpad_tgt= None)
        
        return mask.repeat( self.num_heads, 1, 1)

    def _make_source_mask_seg(self, idx):
        # maskPast: for a single document
        mask = torch.zeros(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]
        
        for i in range(len(sep_tgt)-1 ):
            begin_tgt = leftpad_tgt + sep_tgt[i+1] 
            if i+1 < len(sep_src):
                begin_src = leftpad_src + sep_src[i] 
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to mask also <sep> in previous src sent
                # end_src = leftpad_src + sep_src[i+1] 
                mask[begin_tgt:, begin_src : end_src] = torch.tensor(True)
            else:
                # only </s> is not masked 
                mask[begin_tgt:, leftpad_src : -1 ] = torch.tensor(True)

        # to remove
        # self.test_verbose(idx, mask, leftpad_src, leftpad_tgt)

        return mask.repeat( self.num_heads, 1, 1)


class BatchMaskFuture(BatchMask):
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
        has_incremental_state : bool = False, 
        ):
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            mask_mode = 'future',
            has_incremental_state = has_incremental_state,
            )

    def _make_source_mask_seg_incremental(self, idx):
        # maskFuture: in case of incremental state,  self.tgt_len = 1
        mask = torch.zeros(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]

        # if len(sep_tgt) == len(sep_src), then no future for the current sentence
        if len(sep_tgt) < len(sep_src):
            mask_idx = leftpad_src + sep_src[ len(sep_tgt)] 
            # if don't mask current src <sep>
            # mask_idx = leftpad_src + sep_src[len(sep_tgt)-1] +1 
            mask[ :, mask_idx : ] = torch.tensor(True)

        # to remove
        # self.test_verbose( idx, mask, leftpad_src)

        return mask.repeat( self.num_heads, 1, 1)

    def _make_source_mask_seg(self, idx):
        # maskFuture: for a single document
        mask = torch.zeros(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        if len(sep_tgt) == 1 and len(sep_src) > 1:
            # during inference, when generating the first sentence
            mask[leftpad_tgt: , leftpad_src + sep_src[1] : ] = torch.tensor(True)

        for i in range(len(sep_tgt)-1 ):
            # if i+1 >= len(sep_src), no future token to mask, all tokens are at past
            if i+1 < len(sep_src):
                begin_tgt = leftpad_tgt + sep_tgt[i] # first token or previous tgt <sep>
                end_tgt = leftpad_tgt + sep_tgt[i+1] # +1 mask <sep> with current tgt, != the inference behavior
                
                begin_src = leftpad_src + sep_src[i+1]  # if mask <sep> of current src sentence:
                # begin_src = leftpad_src + sep_src[i+1] +1 
                mask[begin_tgt: end_tgt, begin_src : ] = torch.tensor(True)
        
        # to remove
        # self.test_verbose(idx, mask, leftpad_src, leftpad_tgt)

        return mask.repeat( self.num_heads, 1, 1)
    


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
        has_incremental_state : bool = False, 
        ):
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            mask_mode = 'all',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_source_mask_seg_incremental(self, idx):
        # maskAllSource: in case of incremental state,  self.tgt_len = 1
        mask = torch.ones(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        if leftpad_src > 0:
            mask[:, :leftpad_src] = torch.tensor(False)

        # +1 to mask <sep>, mask nothing if sep_tgt = [0]
        nb_tgt = len(sep_tgt)
        if nb_tgt == 1 and len(sep_src) > 1:
            #only have future
            mask[:, leftpad_src : leftpad_src + sep_src[1]] = torch.tensor(False)
            # print('TEST only future', idx)
            # self.test_verbose( idx, mask, leftpad_src)
            return mask.repeat( self.num_heads, 1, 1)
        
        if nb_tgt  < len(sep_src):
            # past & future
            mask[:, leftpad_src + sep_src[nb_tgt -1]+1 : leftpad_src + sep_src[nb_tgt] ] = torch.tensor(False)
                
        elif nb_tgt == len(sep_src):
            # only have past, no future
            mask[:, leftpad_src + sep_src[-1] + 1 : ] = torch.tensor(False)
        else:
            # in case of len(sep_tgt) > len(sep_src), only </s> is not masked
            mask[:, -1 ] = torch.tensor(False)

        # self.test_verbose(idx, mask, leftpad_src)

        return mask.repeat( self.num_heads, 1, 1)

    def _make_source_mask_seg(self, idx):
        # maskAllSource: for a single document
        mask = torch.ones(self.tgt_len, self.src_len, dtype = torch.bool)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]
        if leftpad_src > 0:
            mask[:, :leftpad_src] = torch.tensor(False)
        if leftpad_tgt > 0:
            mask[:leftpad_tgt, :] = torch.tensor(False)

        if len(sep_tgt) == 1 and len(sep_src) > 1:
            # during inference, when generating the first sentence (only have future)
            mask[leftpad_tgt:, leftpad_src : leftpad_src + sep_src[1]] = torch.tensor(False)
            # print('TEST only future', idx)
            # self.test_verbose(idx, mask, leftpad_src, leftpad_tgt)
            return mask.repeat( self.num_heads, 1, 1)

        for i in range(len(sep_tgt)-1 ):
            begin_tgt = leftpad_tgt + sep_tgt[i] # + 1 if i > 0  else leftpad_tgt 
            end_tgt = leftpad_tgt + sep_tgt[i+1]  # +1  <sep> follows the mask of current sentence 
            if i+1 < len(sep_src):
                begin_src = leftpad_src + sep_src[i] +1 if i > 0  else leftpad_src
                end_src = leftpad_src + sep_src[i+1] 

                mask[begin_tgt : end_tgt, begin_src : end_src] = torch.tensor(False)
            else:
                # in case that len(sep_tgt) > len(sep_src), for last parallel sentence
                begin_src = leftpad_src + sep_src[-1]
                mask[leftpad_tgt + sep_tgt[len(sep_src)-1] : leftpad_tgt + sep_tgt[len(sep_src)], begin_src:] = torch.tensor(False)
                # for supplementary sentence: only </s> is not masked 
                mask[leftpad_tgt + sep_tgt[len(sep_src)] :, -1 ] = torch.tensor(False)

                # print('TEST len_tgt > len_src')
                # self.test_verbose(idx, mask, leftpad_src, leftpad_tgt)
                return mask.repeat( self.num_heads, 1, 1)

        # last sentence (len(sep_tgt) <= len(sep_src) )
        begin_tgt = leftpad_tgt+sep_tgt[-1] # +1 
        if len(sep_tgt) == len(sep_src):
            mask[ begin_tgt:, leftpad_src + sep_src[len(sep_tgt)-1] +1 : ] = torch.tensor(False)
        else:
            mask[ begin_tgt:, leftpad_src + sep_src[len(sep_tgt)-1] +1 : leftpad_src + sep_src[len(sep_tgt)]] = torch.tensor(False)
        
        # to remove
        # self.test_verbose( idx, mask, leftpad_src, leftpad_tgt)
        return mask.repeat( self.num_heads, 1, 1)
