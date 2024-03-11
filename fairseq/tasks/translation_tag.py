# adapted from fairseq/task/translation.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import logging
import os
from typing import Optional
from argparse import Namespace

from fairseq import utils
from fairseq.logging import metrics
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from fairseq.tasks.translation import (
    load_langpair_dataset,
    TranslationConfig,
    TranslationTask,
)

logger = logging.getLogger(__name__)

@dataclass
class TranslationTagConfig(TranslationConfig):
    # TODO ziqian clean the duplicate extra_symbols info in translation.py
    extra_symbols: Optional[str] = field(
        default=None, metadata={"help": "list of extra_special_symbols, separate by single space"}
    )


@register_task("translation_tag", dataclass=TranslationTagConfig)
class TranslationTagTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.
    a version that allow custom special symbol 
    """
    cfg: TranslationTagConfig

    @classmethod
    def load_dictionary(cls, filename, extra_special_symbols=None):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
            extra_special_symbols: extra custom symbol to add
        """
        if extra_special_symbols:
                return Dictionary.load(filename, extra_special_symbols = extra_special_symbols)
        return Dictionary.load(filename)
    
    # @classmethod
    # def load_dictionary(cls, filename, extra_special_symbols=None, extra_symbols_to_end = False):
    #     """Load the dictionary from the filename"""
    #     if extra_special_symbols:
    #         if extra_symbols_to_end: 
    #             d = Dictionary.load(filename)
    #             d.add_extra_symbols(extra_special_symbols)
    #             return d 
    #         else:
    #             return Dictionary.load(filename, extra_special_symbols = extra_special_symbols)
    #     return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8, extra_special_symbols=None #, extra_symbols_to_end = False
    ):
        """Build the dictionary
        rewrite to include extra special symbols
        check fairseq_task.py for the documentation of other argument
        """
        
        d = Dictionary(extra_special_symbols = extra_special_symbols) # if not extra_symbols_to_end and extra_special_symbols else Dictionary()

        for filename in filenames:
            Dictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


    @classmethod
    def setup_task(cls, cfg: TranslationTagConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        #1305 ziqian
        extra_special_symbols = None
        if cfg.extra_symbols:
            extra_special_symbols = [s.strip() for s in cfg.extra_symbols.split() if s.strip() ]
            print('fairseq.tasks.translation: extra_special_symbols=', extra_special_symbols)

        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang)),  extra_special_symbols
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang)), extra_special_symbols
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)


    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                # ignore the extra symbol for bleu 
                # TODO ziqian add an argument to control the symbols to ignore
                extra_symbols_to_ignore = self.tgt_dict.get_extra_symbols()  
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


