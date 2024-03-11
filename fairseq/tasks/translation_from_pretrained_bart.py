# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Optional
from fairseq import utils
from fairseq.data import LanguagePairDataset
from dataclasses import dataclass, field

from . import register_task
# from .translation_tag import TranslationTagConfig, TranslationTagTask, load_langpair_dataset
from .translation import TranslationConfig, TranslationTask, load_langpair_dataset

@dataclass
class TranslationFromPretrainedBARTConfig(TranslationConfig):
    lang_list="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,\
        hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"

    extra_symbols: Optional[str] = field(
        default=None, metadata={"help": "list of extra_special_symbols, separate by single space"}
    )
    langs:str= field(
        default=lang_list, metadata={"help": 'comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.'}
    )
    # metavar='LANG'
    prepend_bos: Optional[bool]=field(
        default=None, metadata={"help": "prepend bos token to each sentence, which matches mBART pretraining"}
        # action='store_true',
    )


@register_task("translation_from_pretrained_bart", dataclass=TranslationFromPretrainedBARTConfig)
class TranslationFromPretrainedBARTTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    cfg: TranslationFromPretrainedBARTConfig

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs',  type=str, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        # fmt: on

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.langs = cfg.langs.split(",")
        print(len(self.langs))
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            d.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=getattr(self.cfg, "max_source_positions", 1024),
            max_target_positions=getattr(self.cfg, "max_target_positions", 1024),
            load_alignments=self.cfg.load_alignments,
            prepend_bos=getattr(self.cfg, "prepend_bos", False),
            append_source_id=True,
        )

    def build_generator(self, models, cfg, **unused):
        if getattr(cfg, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator

            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(cfg, "beam", 5),
                max_len_a=getattr(cfg, "max_len_a", 0),
                max_len_b=getattr(cfg, "max_len_b", 200),
                min_len=getattr(cfg, "min_len", 1),
                normalize_scores=(not getattr(cfg, "unnormalized", False)),
                len_penalty=getattr(cfg, "lenpen", 1),
                unk_penalty=getattr(cfg, "unkpen", 0),
                temperature=getattr(cfg, "temperature", 1.0),
                match_source_len=getattr(cfg, "match_source_len", False),
                no_repeat_ngram_size=getattr(cfg, "no_repeat_ngram_size", 0),
                eos=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index("[{}]".format(self.cfg.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
        return dataset
