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


logger = logging.getLogger(__name__)

def _get_offset(M, l):
    """
        M: predefined maximum length
        l: input length
        return : list of probability for position offset values from 0 to M-L+1
    """
    # l is input length from L_list
    if 2*l >= M:
    # logger.info(f"DEBUG 2*input length = 2*{l} >= M= {M}, M%l = {M%l}")
        return 0
    # M//l possible amount of offset values for each L
    n_skip = torch.randint(0, M//l, (1,)).item()
    # potential offset values for current input length
    offset_values = torch.arange(0, M-l+1, l)
    # logger.info(f"DEBUG original offset_values = {offset_values}")

    offset_values[n_skip:] = offset_values[n_skip:] + M%l
    # logger.info(f"DEBUG updated offset_values = {offset_values}")
    # randomly choose one
    offset = offset_values[ torch.randperm(len(offset_values))[0] ]
    return offset

def get_offset_list(M, L_list):
    # randomly pick one value as offset for each L
    offset_list = L_list.detach().clone()
    offset_list = offset_list.cpu().apply_(lambda x: _get_offset(M, x) ).to(device = L_list.device)
    return offset_list.view(len(offset_list), 1)  

def check_max_len_with_offset(M, L_list, offset_list):
    max_pos_list = L_list.detach().clone() + offset_list.detach().clone()
    if max(max_pos_list) > M :
        # if the position exceed M, return False and the update offset
        diff_list = max_pos_list - M
        diff_list[diff_list < 0] = 0
        offset_list = offset_list - diff_list
        # if a negative value exists in offset_list here, then some L in L_list> M
        # usually this will not happen, and it will be detected later when checking max_positions
        offset_list[offset_list<0] = 0
        return False, offset_list
    return True, offset_list

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    share_pos_offset_enc_dec :bool=True,
    max_source_position: int= None,
    max_target_position: int= None,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    # ziqian 2024-06-10 unifPos offset
    src_offsets = get_offset_list(M = max_source_position, L_list = src_lengths)

    prev_output_tokens = None
    target = None
    tgt_offsets = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()
        # ziqian 2024-06-10 unifPos offset
        if share_pos_offset_enc_dec:
            within_max, src_offsets = check_max_len_with_offset(max_target_position, tgt_lengths, src_offsets)
            if within_max:
                logger.info(f"reset source offsets to share them with target within max_target_position ={max_target_position} ")    
            tgt_offsets = src_offsets

        else:
            tgt_offsets = get_offset_list(M = max_target_position, L_list = tgt_lengths)

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "src_offsets": src_offsets,
            "tgt_offsets": tgt_offsets,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch

# adapt from fairseq/data/language_pair_dataset/LanguagePairDataset
class LanguagePairPosUnifDataset(LanguagePairDataset):
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
            src,
            src_sizes,
            src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
            share_pos_offset_enc_dec: bool =True,
            max_source_position: int= None,
            max_target_position: int= None,

        ):
            super().__init__(
                src,
                src_sizes,
                src_dict,
                tgt,
                tgt_sizes,
                tgt_dict,
                left_pad_source,
                left_pad_target,
                shuffle,
                input_feeding,
                remove_eos_from_source,
                append_eos_to_target,
                align_dataset,
                constraints,
                append_bos,
                eos,
                num_buckets,
                src_lang_id,
                tgt_lang_id,
                pad_to_multiple,
            )
            # compute the document length at sentence-level (e.g. how many sentences are in each doc)
            self.share_pos_offset_enc_dec = share_pos_offset_enc_dec
            self.max_source_position = max_source_position
            self.max_target_position = max_target_position    

    # def __getitem__(self, index):
    #     example = super().__getitem__(index)
    #     # ziqian add position offset
    #     example['src_offset'] = self.src_offset_list[index]
    #     example['tgt_offset'] = self.tgt_offset_list[index] 
    #     return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            share_pos_offset_enc_dec= self.share_pos_offset_enc_dec,
            max_source_position = self.max_source_position,
            max_target_position = self.max_target_position,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res
    


    

