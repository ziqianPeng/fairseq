# ziqian 2024-06-10
# adapt from fairseq/data/language_pair_dataset.py to initialise the random offset that forms uniform probability P(position i is observed)
# register the offset in sample['net_input'] (batch['net_input'])

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.data.multilingual.sampled_multi_dataset import SampledMultiDataset

from fairseq.utils import get_offset_list


logger = logging.getLogger(__name__)


def check_max_len_with_offset(M, L_list, offset_list):
    max_pos_list = L_list.detach().clone() + offset_list.detach().clone() #.view(len(offset_list))

    if max(max_pos_list) > M :
        # if the position exceed M, return False and the update offset
        diff_list = max_pos_list - M
        diff_list[diff_list < 0] = 0
        # offset_list = offset_list.view(len(offset_list)) - diff_list
        offset_list = offset_list - diff_list
        # if a negative value exists in offset_list here, then some L in L_list> M
        # usually this will not happen, and it will be detected later when checking max_positions
        offset_list[offset_list<0] = 0
        return False, offset_list
    return True, offset_list



# adapt from fairseq/data/transform_eos_lang_pair_dataset.py/TransformEosLangPairDataset
class PosUnifDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.
    to initialise the random offset that forms uniform probability P(position i is observed)
    register the offset in sample['net_input'] (batch['net_input'])

    Args:
        share_pos_offset_enc_dec : whether share the same position offset in encoder and decoder
        max_source_position: max position for the source language (e.g. in source position embedding)
        max_target_position: max position for the target language 

    check other argument at fairseq/data/language_pair_dataset/LanguagePairDataset

    """
    
    def __init__(
        self,
        split: str,
        dataset: FairseqDataset,
        share_pos_offset_enc_dec: bool= True,
        max_source_position: int= None,
        max_target_position: int= None,
        deactive_pos_unif: bool = True,
        pad_idx:int = None,
        ):

        self.dataset = dataset
        # compute the document length at sentence-level (e.g. how many sentences are in each doc)
        self.share_pos_offset_enc_dec = share_pos_offset_enc_dec
        self.max_source_position = max_source_position
        self.max_target_position = max_target_position 
        self.split = split
        self.deactive_pos_unif = deactive_pos_unif
        self.pad_idx = pad_idx

        # todo ziqian add assertion to make sure that the id order is 0,1,2,...
        # logger.info(f" init order of iteration {torch.LongTensor(list(map(lambda x: x['id'], dataset)))}")

        src_offsets, tgt_offsets = self.reset_offsets(dataset)
        self.src_offset_list = src_offsets
        self.tgt_offset_list = tgt_offsets


    def __getitem__(self, index):
        example = self.dataset[index]
        example['src_offsets'] = self.src_offset_list[index] if self.src_offset_list is not None else None
        example['tgt_offsets'] = self.tgt_offset_list[index] if self.tgt_offset_list is not None else None
        return example

    def __len__(self):
        return len(self.dataset)
    
    def get_source_offsets(self):
        return self.src_offset_list
    
    def get_target_offsets(self):
        return self.tgt_offset_list
    
    def _get_offsets_sample_source(self, item):
        # error if self.dataset is sampled_multi_dataset
        # logger.info(f"DEBUG dataset item src = {item}")

        src_lengths = torch.LongTensor([item["source"].ne(self.pad_idx).long().sum()])
        src_offsets = get_offset_list(M = self.max_source_position, L_list = src_lengths)

        if item.get("target", None) is not None: 
            tgt_lengths = torch.LongTensor(item["target"].ne(self.pad_idx).long().sum())

            if self.share_pos_offset_enc_dec:
                within_max, src_offsets = check_max_len_with_offset(self.max_target_position, tgt_lengths, src_offsets)    

        return src_offsets
    

    def _get_offsets_sample_target(self, item):
        assert(item.get("target", None) is not None)

        tgt_lengths = torch.LongTensor([item["target"].ne(self.pad_idx).long().sum()])
        tgt_offsets = get_offset_list(M = self.max_target_position, L_list = tgt_lengths)
        # tgt_offsets = get_offset_list(M = 2048, L_list = tgt_lengths)

        return tgt_offsets
    

    def reset_offsets(self, dataset):
        """
        """ 
        if self.split != 'train' and self.deactive_pos_unif:
            return None, None
        
        logger.info( f"reset_offsets split {self.split}")
        src_offsets = torch.cat(list(map(self._get_offsets_sample_source, dataset))).long()
        tgt_offsets = None

        if dataset[0].get("target", None) is not None: 
            if self.share_pos_offset_enc_dec:
                tgt_offsets = src_offsets.detach().clone()
            else:
                tgt_offsets = torch.cat(list(map(self._get_offsets_sample_target, dataset))).long()

        # logger.info(f"DEBUG src_offsets {len(src_offsets)} {src_offsets}")
        # logger.info(f"DEBUG tgt_offsets {len(tgt_offsets)} {tgt_offsets}")
        return src_offsets, tgt_offsets
    

    def collater(self, samples, **extra_args):
        """Merge a list of samples to form a mini-batch.

        Args:

        check other argument at fairseq/data/language_pair_pos_unif_dataset/LanguagePairPosUnifDataset

        """
        # TODO ziqian 1. this will run at each epoch, rewrite to decide offset list in init and 
        # make it accessible from get_items() thus collater only take the value
        # 2. in models deactive pos unif if offset of valid and test is None, instead of control with the var deactive_pos_unif
        # logger.info(f"DEBUG collater samples = {samples}")
        batch = self.dataset.collater(samples, **extra_args)
        if len(batch) == 0 or "net_input" not in batch:
            return batch
        
        # logger.info(f"DEBUG collater batch = {batch}")
        if "src_offsets" in batch["net_input"] and "tgt_offsets" in batch["net_input"]:
            # "tgt_offsets" can be None
            return batch
        
        src_offsets = None if samples[0].get('src_offsets', None) is None else torch.LongTensor(list(map(lambda x: self.src_offset_list[x], batch['id'])))
        tgt_offsets = None if samples[0].get('tgt_offsets', None) is None else torch.LongTensor(list(map(lambda x: self.tgt_offset_list[x], batch['id'])))

        # src_ids = 
        batch["net_input"]["src_offsets"]= src_offsets
        batch["net_input"]["tgt_offsets"]= tgt_offsets
        # logger.info(f'DEBUG final src_offsets = {src_offsets}')
        # logger.info(f'DEBUG final tgt_offsets = {tgt_offsets}')

        # src_offsets = get_offset_list(M = self.max_source_position, L_list = batch["net_input"]["src_lengths"])
        # # logger.info(f'DEBUG src_offsets = {src_offsets}')
        # tgt_offsets = None
        # if batch.get("target", None) is not None:
        #     # has target
        #     tgt_lengths = torch.LongTensor( batch['target'].ne(1).long().sum(axis = 1))
        #     # tgt_lengths = torch.LongTensor(
        #     #     [s["target"].ne(self.pad_idx).long().sum() for s in batch])

        #     if self.split == 'train' or not self.deactive_pos_unif:
        #         if self.share_pos_offset_enc_dec:
        #             within_max, src_offsets = check_max_len_with_offset(self.max_target_position, tgt_lengths, src_offsets)    
        #             tgt_offsets = src_offsets.detach().clone()
        #         else:
        #             tgt_offsets = get_offset_list(M = self.max_target_position, L_list = tgt_lengths)
       
        # batch["net_input"]["src_offsets"]= src_offsets if self.split == 'train' or not self.deactive_pos_unif else None
        # batch["net_input"]["tgt_offsets"]= tgt_offsets
        # logger.info(f'DEBUG final src_offsets = {src_offsets}')
        
        return batch
    
    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
    
    @property
    def sizes(self):
        return self.dataset.sizes
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return getattr(self.dataset, "can_reuse_epoch_itr_across_epochs", False)
    
    def set_epoch(self, epoch):
        return self.dataset.set_epoch(epoch)
    
    

