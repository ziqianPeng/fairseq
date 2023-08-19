# adapt from batch_mask.py 

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
        factor_mode : str = 'all',
        has_incremental_state : bool = False, 
        ):
        """
        fade_mode:  apply attention factor to past, future or all source context 
        """
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
        self.factor_base = factor_base
        self.factor_mode = factor_mode
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
            batch_factor = list(map(self._make_attn_factor_seg_incremental, range(len(self.sep_idx_tgt))))
        else:
            batch_factor = list(map(self._make_attn_factor_seg, range(len(self.sep_idx_tgt))))

        return torch.cat(batch_factor).to(self.sep_idx_tgt.device)
    
    def test_verbose(self, idx, factor, leftpad_src, leftpad_tgt = None):
        if self.src_len < 30 and self.sep_idx_tgt.size(1) > 1 and idx < 4:
            print(f'TEST idx {idx}, src_len { self.src_len} tgt_len {self.tgt_len}')
            print(f'TEST src_leftpad { leftpad_src}')
            if leftpad_tgt:
                print(f'TEST tgt_leftpad {leftpad_tgt}')
            print('TEST sep_idx_src', self.sep_idx_src)
            print('TEST sep_idx_tgt', self.sep_idx_tgt)
            print('TEST attention factor')
            for i in range(len(factor)):
                print(i, factor[i])


############################################################
############################################################
class BatchAttnFactorSentPast(BatchAttnFactor):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    
      factor_type:
          1. 10**(-d)
          2. (90%)*(d)
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
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            factor_base=factor_base,
            factor_mode='past',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_attn_factor_seg_incremental(self, idx):
        # fadePast:  in case of incremental state,  self.tgt_len = 1
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   

        if len(sep_tgt) > len(sep_src):
            # only </s> is not masked if more target segments are generated
            factor_matrix[:, leftpad_src : -1 ] = self.factor_base

            # self.test_verbose( idx, factor_matrix, leftpad_src)
            return factor_matrix.repeat( self.num_heads, 1, 1)
            
        if len(sep_tgt) > 1:
            # +1 to mask <sep>, mask nothing if sep_tgt = [0] (no past)
            mask_idx = leftpad_src + sep_src[len(sep_tgt)-1] 
            factor_matrix[:, leftpad_src : mask_idx + 1] = self.factor_base

        # self.test_verbose( idx, factor_matrix, leftpad_src)
        return factor_matrix.repeat( self.num_heads, 1, 1)


    def _make_attn_factor_seg(self, idx):
        # fadePast: for a single document
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
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
                factor_matrix[begin_tgt:, begin_src : end_src] = self.factor_base
            else:
                # only </s> is not masked 
                factor_matrix[begin_tgt:, leftpad_src : -1 ] = self.factor_base

        # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
        return factor_matrix.repeat( self.num_heads, 1, 1)



class BatchAttnFactorSentFuture(BatchAttnFactor):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    
      factor_type:
          1. (0.1)**(d)
          2. (0.9)*(d)
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
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            factor_base=factor_base,
            factor_mode='future',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_attn_factor_seg_incremental(self, idx):
        # fadePast:  in case of incremental state,  self.tgt_len = 1
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   
        
        # if len(sep_tgt) == len(sep_src), then no future for the current sentence
        if len(sep_tgt) < len(sep_src):
            mask_idx = leftpad_src + sep_src[ len(sep_tgt)] 
            factor_matrix[ :, mask_idx :  ] = self.factor_base

        # self.test_verbose( idx, factor_matrix, leftpad_src)
        return factor_matrix.repeat( self.num_heads, 1, 1)


    def _make_attn_factor_seg(self, idx):
        # fadePast: for a single document
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        if len(sep_tgt) == 1 and len(sep_src) > 1:
            # during inference, when generating the first sentence
            factor_matrix[leftpad_tgt: , leftpad_src + sep_src[1]  :  ] = self.factor_base

        for i in range(len(sep_tgt)-1 ):
            # if i+1 >= len(sep_src), no future token to mask, all tokens are at past
            if i+1 < len(sep_src):
                begin_tgt = leftpad_tgt + sep_tgt[i] # first token or previous tgt <sep>
                end_tgt = leftpad_tgt + sep_tgt[i+1] # +1 mask <sep> with current tgt, != the inference behavior
                
                begin_src = leftpad_src + sep_src[i+1]  # mask <sep> of current src sentence:
                factor_matrix[begin_tgt: end_tgt, begin_src :  ] =  self.factor_base

        # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
        return factor_matrix.repeat( self.num_heads, 1, 1)


class BatchAttnFactorSentAll(BatchAttnFactor):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    
      factor_type:
          1. (0.1)**(d)
          2. (0.9)*(d)
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
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            factor_base=factor_base,
            factor_mode='all',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_attn_factor_seg_incremental(self, idx):
        # fadePast:  in case of incremental state,  self.tgt_len = 1
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   

        # +1 to mask <sep>, mask nothing if sep_tgt = [0]
        nb_tgt = len(sep_tgt)
        if nb_tgt == 1 and len(sep_src) > 1:
            #only have future
            for i in range(len(sep_src) -1 ):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[:, begin_src : ] = self.factor_base
            
            # self.test_verbose( idx, factor_matrix, leftpad_src)
            return factor_matrix.repeat( self.num_heads, 1, 1)
        
        if nb_tgt  <= len(sep_src):
            # past 
            for i in range(nb_tgt -1 ):
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to include <sep> in previous src sent
                factor_matrix[:, leftpad_src : end_src] =  self.factor_base
            # future
            for i in range(nb_tgt-1, len(sep_src)-1):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[:, begin_src : ] =  self.factor_base
                
        else:
            # len(sep_tgt) > len(sep_src), reduce attention of all tokens except </s>
            factor_matrix[:, -1 ] = self.factor_base

        # self.test_verbose( idx, factor_matrix, leftpad_src)
        return factor_matrix.repeat( self.num_heads, 1, 1)


    def _make_attn_factor_seg(self, idx):
        # fadePast: for a single document
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        if len(sep_tgt) == 1 and len(sep_src) > 1:
            # during inference, when generating the first sentence (only have future)
            for i in range(len(sep_src) -1 ):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[leftpad_tgt:, begin_src : ] = self.factor_base

            # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
            return factor_matrix.repeat( self.num_heads, 1, 1)
        
        # past
        for i in range(len(sep_tgt)-1 ):
            begin_tgt = leftpad_tgt + sep_tgt[i+1] 
            if i+1 < len(sep_src):
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to mask also <sep> in previous src sent
                factor_matrix[begin_tgt:, leftpad_src : end_src] = self.factor_base
            else:
                # only </s> is still 1, reduce attention of all other positions 
                factor_matrix[begin_tgt:, leftpad_src : -1 ] =  self.factor_base
        # future
        for i in range(len(sep_tgt)-1, 0, -1 ):
            # if i >= len(sep_src), no future token to be faded, all tokens are at past
            if i < len(sep_src):
                # begin_tgt = leftpad_tgt + sep_tgt[i] # first token or previous tgt <sep>
                end_tgt = leftpad_tgt + sep_tgt[i] # +1 include <sep> with current tgt, != the inference behavior                
                begin_src = leftpad_src + sep_src[i]  # if include <sep> of current src sentence:

                factor_matrix[leftpad_tgt: end_tgt, begin_src:] = self.factor_base
        
        # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
        return factor_matrix.repeat( self.num_heads, 1, 1)

##########################################################################
############################## FADE ######################################
##########################################################################


class BatchAttnFadeSentPast(BatchAttnFactor):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    
      factor_type:
          1. 10**(-d)
          2. (90%)*(d)
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
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            factor_base=factor_base,
            factor_mode='past',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_attn_factor_seg_incremental(self, idx):
        # fadePast:  in case of incremental state,  self.tgt_len = 1
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   
        
        if len(sep_tgt) > len(sep_src):
            # reduce attention of all other positions except </s> if more target segments are generated
            factor_matrix[:, leftpad_src : -1 ] = self.factor_base**(len(sep_tgt))

            # self.test_verbose( idx, factor_matrix, leftpad_src)
            return factor_matrix.repeat( self.num_heads, 1, 1)
            
        if len(sep_tgt) > 1:
            # +1 to mask <sep>, mask nothing if sep_tgt = [0] (no past)
            for i in range(len(sep_tgt)-1 ):
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to include <sep> in previous src sent
                factor_matrix[:, leftpad_src : end_src] = factor_matrix[:, leftpad_src : end_src] * self.factor_base
        
        # self.test_verbose( idx, factor_matrix, leftpad_src)
        return factor_matrix.repeat( self.num_heads, 1, 1)


    def _make_attn_factor_seg(self, idx):
        # fadePast: for a single document
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]
        
        for i in range(len(sep_tgt)-1 ):
            begin_tgt = leftpad_tgt + sep_tgt[i+1] 
            if i+1 < len(sep_src):
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to mask also <sep> in previous src sent
                factor_matrix[begin_tgt:, leftpad_src : end_src] = factor_matrix[begin_tgt:, leftpad_src : end_src] * self.factor_base
            else:
                # only </s> is still 1, reduce attention of all other positions 
                factor_matrix[begin_tgt:, leftpad_src : -1 ] =  self.factor_base**(len(sep_tgt))

        # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
        return factor_matrix.repeat( self.num_heads, 1, 1)


class BatchAttnFadeSentFuture(BatchAttnFactor):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    
      factor_type:
          1. (0.1)**(d)
          2. (0.9)*(d)
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
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            factor_base=factor_base,
            factor_mode='future',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_attn_factor_seg_incremental(self, idx):
        # fadePast:  in case of incremental state,  self.tgt_len = 1
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   

        # if len(sep_tgt) == len(sep_src), then no future for the current sentence
        if len(sep_tgt) < len(sep_src): 
            for i in range(len(sep_tgt)-1, len(sep_src)-1 ):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[:, begin_src : ] = factor_matrix[:, begin_src:] * self.factor_base
        
        # self.test_verbose( idx, factor_matrix, leftpad_src)
        return factor_matrix.repeat( self.num_heads, 1, 1)


    def _make_attn_factor_seg(self, idx):
        # fadePast: for a single document
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        if len(sep_tgt) == 1 and len(sep_src) > 1:
            # during inference, when generating the first sentence
            for i in range(len(sep_src)-1):
                begin_src = leftpad_src + sep_src[i+1] 
                factor_matrix[leftpad_tgt: , begin_src:] = factor_matrix[leftpad_tgt: , begin_src:] * self.factor_base

        for i in range(len(sep_tgt)-1, 0, -1 ):
            # if i >= len(sep_src), no future token to be faded, all tokens are at past
            if i < len(sep_src):
                # begin_tgt = leftpad_tgt + sep_tgt[i] # first token or previous tgt <sep>
                end_tgt = leftpad_tgt + sep_tgt[i] # +1 include <sep> with current tgt, != the inference behavior                
                begin_src = leftpad_src + sep_src[i]  # if include <sep> of current src sentence:

                factor_matrix[leftpad_tgt: end_tgt, begin_src:] = factor_matrix[leftpad_tgt: end_tgt, begin_src:] * self.factor_base
        
        # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
        return factor_matrix.repeat( self.num_heads, 1, 1)


class BatchAttnFadeSentAll(BatchAttnFactor):
    """
    size of attention mask : (bsz * num_heads, tgt_len, src_len) or (tgt_len, src_len)
    
      factor_type:
          1. (0.1)**(d)
          2. (0.9)*(d)
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
        super().__init__(
            sep_idx_src = sep_idx_src,
            sep_idx_tgt = sep_idx_tgt,
            src_len = src_len,
            tgt_len = tgt_len,
            pad_idx_sep = pad_idx_sep,
            pad_left_src = pad_left_src,
            pad_left_tgt = pad_left_tgt,
            num_heads = num_heads,
            factor_base=factor_base,
            factor_mode='all',
            has_incremental_state = has_incremental_state,
            )

    
    def _make_attn_factor_seg_incremental(self, idx):
        # fadePast:  in case of incremental state,  self.tgt_len = 1
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]   

        # +1 to mask <sep>, mask nothing if sep_tgt = [0]
        nb_tgt = len(sep_tgt)
        if nb_tgt == 1 and len(sep_src) > 1:
            #only have future
            for i in range(len(sep_src) -1 ):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[:, begin_src : ] = factor_matrix[:, begin_src:] * self.factor_base
            
            # self.test_verbose( idx, factor_matrix, leftpad_src)
            return factor_matrix.repeat( self.num_heads, 1, 1)
        
        if nb_tgt  <= len(sep_src):
            # past 
            for i in range(nb_tgt -1 ):
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to include <sep> in previous src sent
                factor_matrix[:, leftpad_src : end_src] = factor_matrix[:, leftpad_src : end_src] * self.factor_base
            # future
            for i in range(nb_tgt-1, len(sep_src)-1):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[:, begin_src : ] = factor_matrix[:, begin_src:] * self.factor_base
                
        else:
            # len(sep_tgt) > len(sep_src), reduce attention of all tokens except </s>
            factor_matrix[:, -1 ] = self.factor_base**(len(sep_tgt))

        # self.test_verbose( idx, factor_matrix, leftpad_src)
        return factor_matrix.repeat( self.num_heads, 1, 1)


    def _make_attn_factor_seg(self, idx):
        # fadePast: for a single document
        factor_matrix = torch.ones(self.tgt_len, self.src_len)
        # remove padding for self.sep_idx_tgt[idx] and self.sep_idx_src[idx] 
        sep_tgt = self.sep_idx_tgt[idx][ self.sep_idx_tgt[idx].ne(self.pad_idx_sep) ] 
        sep_src = self.sep_idx_src[idx][ self.sep_idx_src[idx].ne(self.pad_idx_sep) ] 
        # assert len(sep_tgt) == len(sep_src) # False during generation
        leftpad_src = 0 if self.pad_left_src is None else self.pad_left_src[idx]
        leftpad_tgt = 0 if self.pad_left_tgt is None else self.pad_left_tgt[idx]

        if len(sep_tgt) == 1 and len(sep_src) > 1:
            # during inference, when generating the first sentence (only have future)
            for i in range(len(sep_src) -1 ):
                begin_src = leftpad_src + sep_src[i+1] # +1 to exlude <sep> in current src sent
                factor_matrix[leftpad_tgt:, begin_src : ] = factor_matrix[leftpad_tgt:, begin_src:] * self.factor_base

            # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
            return factor_matrix.repeat( self.num_heads, 1, 1)
        
        # past
        for i in range(len(sep_tgt)-1 ):
            begin_tgt = leftpad_tgt + sep_tgt[i+1] 
            if i+1 < len(sep_src):
                end_src = leftpad_src + sep_src[i+1] +1 # +1 to mask also <sep> in previous src sent
                factor_matrix[begin_tgt:, leftpad_src : end_src] = factor_matrix[begin_tgt:, leftpad_src : end_src] * self.factor_base
            else:
                # only </s> is still 1, reduce attention of all other positions 
                factor_matrix[begin_tgt:, leftpad_src : -1 ] =  self.factor_base**(len(sep_tgt))

        # future
        for i in range(len(sep_tgt)-1, 0, -1 ):
            # if i >= len(sep_src), no future token to be faded, all tokens are at past
            if i < len(sep_src):
                # begin_tgt = leftpad_tgt + sep_tgt[i] # first token or previous tgt <sep>
                end_tgt = leftpad_tgt + sep_tgt[i] # +1 include <sep> with current tgt, != the inference behavior                
                begin_src = leftpad_src + sep_src[i]  # if include <sep> of current src sentence:

                factor_matrix[leftpad_tgt: end_tgt, begin_src:] = factor_matrix[leftpad_tgt: end_tgt, begin_src:] * self.factor_base
        
        # self.test_verbose( idx, factor_matrix, leftpad_src, leftpad_tgt)
        return factor_matrix.repeat( self.num_heads, 1, 1)
