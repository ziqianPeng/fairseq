# adapt from fairseq/data/language_pair_dataset.py

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset


def _get_sep_info(item, sep_idx = 4):
    sep_indice = torch.cat([ torch.tensor([0]),torch.nonzero(item == sep_idx).view(-1)]).long()
    return sep_indice

def doc_size_in_sents(dataset):
    """
    usually run once to init LanguagePairDocDataset 
    Returns:
        sep_info (list of LongTensor): list of sep indices for each document, always begin by 0
        nb_sents (LongTensor): number of segments separated by sep tag for each document 
    """
    sep_info = list(map(_get_sep_info, dataset))
    nb_sents = torch.LongTensor(list(map(lambda x: len(x), sep_info)))
    return sep_info, nb_sents


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """
    adapt from fairseq/data/language_pair_dataset/collate
    sort by number of sentence, then for document with the same #sent, sort by #token
    samples are selected samples to form a batch

    samples contain 'sep_idx_src' and 'sep_idx_tgt', the indice of <sep> tag in each source/target sample
    samples prepared in __getitems() of DataSet class below
    """
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
    #############
    # TODO ziqian test it
    # 1.sort by number of sentences  
    src_stat = [
        [s["sep_idx_src"].size()[0], s["source"].ne(pad_idx).long().sum().item()] for s in samples]    

    # 2.sort by descending source length for document with the same number of sentences
    sort_order = torch.LongTensor(
        sorted(range(len(src_stat)), key=lambda k: (src_stat[k][0], src_stat[k][1]), reverse = True)
    )
    src_lengths = torch.LongTensor(src_stat)[:, 1]

    # 3. store parallel sep indice (src)
    # TODO ziqian, check which one is better (left_pad/ right_pad)
    sep_indice_src = merge(
        "sep_idx_src",
        left_pad=left_pad_source,
        pad_to_length=None,
    )
    sep_indice_src = sep_indice_src.index_select(0, sort_order)

    #######
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    sep_indice_tgt = None
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

        #### 3. store parallel sep indice (tgt)
        # TODO ziqian, check which one is better (left_pad/ right_pad)
        sep_indice_tgt = merge(
            "sep_idx_tgt",
            left_pad=left_pad_source,
        )
        sep_indice_tgt = sep_indice_tgt.index_select(0, sort_order)
        ####

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
        },
        "target": target,
        "sep_idx_src": sep_indice_src,
        "sep_idx_tgt": sep_indice_tgt,
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
class LanguagePairDocSimpleDataset(LanguagePairDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src_nb_sents = nb_sents
        src_sep_indice = sep_info

    add <sep> index of source text for every batch
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
            
            sep_info, nb_sents = doc_size_in_sents(src)
            self.src_nb_sents = nb_sents
            self.src_sep_indice = sep_info
            
            sep_info, nb_sents = doc_size_in_sents(tgt)
            self.tgt_nb_sents = nb_sents
            self.tgt_sep_indice = sep_info


    def __getitem__(self, index, sep_idx = 4):
        """
        adapt from fairseq, to add sep index
        sep_idx is 4 by default
        """
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        # ziqian add sep index
        sep_indice_src = self.src_sep_indice[index]
        sep_indice_tgt = self.tgt_sep_indice[index] if self.tgt is not None else None

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "sep_idx_src": sep_indice_src, 
            "sep_idx_tgt": sep_indice_tgt, 
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example


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
            #### code from fairseq, but call collate defined above ####
            res = collate(
                samples,
                pad_idx=self.src_dict.pad(),
                eos_idx=self.eos,
                left_pad_source=self.left_pad_source,
                left_pad_target=self.left_pad_target,
                input_feeding=self.input_feeding,
                pad_to_length=pad_to_length,
                pad_to_multiple=self.pad_to_multiple,
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

    def ordered_indices(self):
        """
        rewrite to sort first on #sentence then on #token for doc with the same #sentences
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            # this helps for example when all segments have length 10
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]
