from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch
from fairseq.data import Dictionary

# for debug use
import logging
logger = logging.getLogger(__name__)

class EncoderLangtok(Enum):
    """
    Prepend to the beginning of source sentence either the
    source or target language token. (src/tgt).
    """

    src = "src"
    tgt = "tgt"


class LangTokSpec(Enum):
    main = "main"
    mono_dae = "mono_dae"


class LangTokStyle(Enum):
    multilingual = "multilingual"
    mbart = "mbart"


@torch.jit.export
def get_lang_tok(
    lang: str, lang_tok_style: str, spec: str = LangTokSpec.main.value
) -> str:
    # TOKEN_STYLES can't be defined outside this fn since it needs to be
    # TorchScriptable.
    TOKEN_STYLES: Dict[str, str] = {
        LangTokStyle.mbart.value: "[{}]",
        LangTokStyle.multilingual.value: "__{}__",
    }

    if spec.endswith("dae"):
        lang = f"{lang}_dae"
    elif spec.endswith("mined"):
        lang = f"{lang}_mined"
    style = TOKEN_STYLES[lang_tok_style]
    return style.format(lang)

def augment_dictionary(
    dictionary: Dictionary,
    language_list: List[str],
    lang_tok_style: str,
    langtoks_specs: Sequence[str] = (LangTokSpec.main.value,),
    extra_data: Optional[Dict[str, str]] = None,
    extra_symbols_to_end: Optional[List[str]] = None,
) -> None:
    nb = 0
    for spec in langtoks_specs:
        for language in language_list:
            # logger.info(f'DEBUG...nb={nb}, len(dic) = {len(dictionary)}')
            dictionary.add_symbol(
                get_lang_tok(lang=language, lang_tok_style=lang_tok_style, spec=spec)
            )
            nb+=1
            # logger.info(f'DEBUG...get{len(dictionary)-1}={dictionary[len(dictionary)-1]}')
            # logger.info(f'DEBUG...{spec}...{language}...{get_lang_tok(lang=language, lang_tok_style=lang_tok_style, spec=spec)}')

    if lang_tok_style == LangTokStyle.mbart.value or (
        extra_data is not None and LangTokSpec.mono_dae.value in extra_data
    ):
        dictionary.add_symbol("<mask>")
    # if extra_symbols_to_end:
    #     logger.info(f'DEBUG len(dict) = {len(dictionary)}')
    #     # if "<mask>" not in dictionary:
    #     #     logger.info(f'Adding symbol <mask> in the dictionary at position {len(dictionary)}')
    #     #     dictionary.add_symbol("<mask>")
    #     # for s in extra_symbols_to_end:
    #     #     logger.info(f'Adding extra special symbol {s} at position {len(dictionary)}')
    #     #     dictionary.add_symbol(s)


