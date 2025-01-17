# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import time

import torch
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    ListDataset,
    data_utils,
    iterators,
    encoders, # eval_bleu
    LanguagePairPosUnifDataset,
    PosUnifDataset,
)
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.utils import FileContentsAction

### eval_bleu
import json
from argparse import Namespace
from fairseq.logging import metrics
import numpy as np
from fairseq import utils
# from fairseq.utils import safe_hasattr
from fairseq.optim.amp_optimizer import AMPOptimizer

EVAL_BLEU_ORDER = 4

###
def get_time_gap(s, e):
    return (
        datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)
    ).__str__()
###


logger = logging.getLogger(__name__)


@register_task("translation_multi_simple_epoch_pos_unif")
class TranslationMultiSimpleEpochPosUnifTask(TranslationMultiSimpleEpochTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')
        parser.add_argument('--extra-symbols-to-end', default=None, 
            help="list of extra_special_symbols to append in the dictionary, separate by single space")
        
        # Ziqian 2024-03-27 position offset
        parser.add_argument('--offset', type = int, default=0, 
            help="position index offset of input sequence, default is 0, so the input position begins at 0 + padding_idx"
            )
        
        # Ziqian 2024-05-15 uniform position
        parser.add_argument('--active-uniform-pos', action='store_true', default=None, 
            help="activate or not uniform position indices using offset during training"
            )
        
        # Ziqian 2024-05-15 uniform position
        parser.add_argument('--active-uniform-pos-inference', action='store_true', default=False, 
            help="activate or not uniform position indices using offset during inference"
            )
        
        parser.add_argument('--share-pos-offset-enc-dec', action='store_true', default=False, 
            help="share the same position offset in encoder and decoder"
            )

        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)


        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores'
                            )
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options'
                                 )
        parser.add_argument('--eval-bleu-detok-args', type=str, default="{}", metavar='JSON',
                            help='args for building the tokenizer, if needed'
                            )
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu'
                            )
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU'
                            )
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\''
                                 )
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation'
                            )
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)

        logger.info(f'position offset = {args.offset}')
        logger.info(f'active_uniform_pos = {args.active_uniform_pos}')
        logger.info(f'share_pos_offset_enc_dec = {args.share_pos_offset_enc_dec}')
        logger.info(f'active_uniform_pos_inference = {args.active_uniform_pos_inference}')

        self.offset = args.offset
        self.active_uniform_pos = args.active_uniform_pos
        self.share_pos_offset_enc_dec = args.share_pos_offset_enc_dec
        self.active_uniform_pos_inference = args.active_uniform_pos_inference


    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        # dataset = LanguagePairPosUnifDataset(src_data, src_lengths, self.source_dictionary)

        src_langtok_spec, tgt_langtok_spec = self.args.langtoks["main"]
        # 2023-11-20 ziqian: enable inference at validation step, when source_lang or target_lang is not given in training arguments
        source_lang = self.source_langs[0] if self.args.source_lang is None else self.args.source_lang 
        target_lang = self.target_langs[0] if self.args.target_lang is None else self.args.target_lang 
        logger.info(f'Buiding dataset for inference with source language {source_lang} and target language {target_lang}...')

        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                dataset,
                src_eos=self.source_dictionary.eos(),
                src_lang=source_lang,
                tgt_eos=self.target_dictionary.eos(),
                tgt_lang=target_lang,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                source_lang,
                target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
            )
        if self.active_uniform_pos:
            dataset = PosUnifDataset(
                dataset,
                share_pos_offset_enc_dec = self.share_pos_offset_enc_dec,
                max_source_position = self.args.max_source_positions,
                max_target_position = self.args.max_target_positions,
                pad_idx = self.source_dictionary.pad(),
                )
        return dataset

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):

        if hasattr(self.args, 'active_uniform_pos_inference') and self.args.active_uniform_pos_inference :
            extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
            extra_gen_cls_kwargs['active_uniform_pos_inference'] = self.args.active_uniform_pos_inference
            logger.info(f"WARNING: set generator's active_uniform_pos_inference as {self.args.active_uniform_pos_inference}")

        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs, 
            prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        )
    
    def build_model(self, args, from_checkpoint=False):
        # adapt from build_model of TranslationTask
        # the args here is cfg.model
        if hasattr(self.args, 'max_source_positions') and self.args.max_source_positions != args.max_source_positions:
            args.max_source_positions = self.args.max_source_positions
            logger.info(f"WARNING: set model's max_source_position as {self.args.max_source_positions}")
        
        if hasattr(self.args, 'max_target_positions') and self.args.max_target_positions != args.max_target_positions:
            args.max_target_positions = self.args.max_target_positions
            logger.info(f"WARNING: set model's max_target_position as {self.args.max_target_positions}")

        if self.args.offset != getattr(args, 'offset', 0) :
            logger.info(f"WARNING: changing model's position offset from {getattr(args, 'offset', None)} to {self.args.offset}")
            setattr(args, 'offset', self.get_offset() )

        if self.args.active_uniform_pos != getattr(args, 'active_uniform_pos', None):
            logger.info(f"TEST: changing model's active_uniform_pos param from {getattr(args, 'active_uniform_pos', None)} to {self.args.active_uniform_pos}")
            setattr(args, 'active_uniform_pos', self.args.active_uniform_pos )

        model = super().build_model(args, from_checkpoint)
                
        if self.args.eval_bleu:
            detok_args = json.loads(self.args.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model


