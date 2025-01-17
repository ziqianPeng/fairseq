# Fine-tuning MBART Models Using BLEU for Checkpoint Selection 


This branch is adapted from the [fairseq](https://github.com/facebookresearch/fairseq) framework, for the purpose of fine-tuning mBART models using BLEU scores to select the best checkpoint for the machine translation task.

The essential modifications are in 
```
fairseq/tasks/translation_multi_simple_epoch_TranslationTask.py
```

Here is an example to run the code:

1. Preprocess the dataset, check `example_scripts/preprocess-encode.sh`
2. Train or fine-tuning the model with the prepared dataset, check `example_scripts/mBART50_FT_scipar.sh`

These two scripts are used to fine-tune [mBART50-one-to-many](https://github.com/facebookresearch/fairseq/blob/main/examples/multilingual/README.md) on the [SciPar](https://elrc-share.eu/repository/browse/scipar-a-collection-of-parallel-corpora-from-scientific-abstracts-v-2021-in-tmx-format/aaf503c0739411ec9c1a00155d02670665aacff53a8543938cd99da54fdd66af/) (Roussis et al. 2022) corpora for the translation from English to French.
The resulting checkpoint is appiled as baseline model in the following publications:

```
@inproceedings{coling2025trad,
    title = {{Towards the Machine Translation of Scientific Neologisms}},
	author={Lerner, Paul and Yvon, François},
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    year = "2025",
    url={https://hal.science/hal-04835653},
    publisher = "International Committee on Computational Linguistics"
}
```

```
@inproceedings{lerner:hal-04623021,
  TITLE = {{Vers la traduction automatique des n{\'e}ologismes scientifiques}},
  AUTHOR = {Lerner, Paul and Yvon, Fran{\c c}ois},
  URL = {https://inria.hal.science/hal-04623021},
  BOOKTITLE = {{35{\`e}mes Journ{\'e}es d'{\'E}tudes sur la Parole (JEP 2024) 31{\`e}me Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles (TALN 2024) 26{\`e}me Rencontre des {\'E}tudiants Chercheurs en Informatique pour le Traitement Automatique des Langues (RECITAL 2024)}},
  ADDRESS = {Toulouse, France},
  EDITOR = {BALAGUER and Mathieu and BENDAHMAN and Nihed and HO-DAC and Lydia-Mai and MAUCLAIR and Julie and MORENO and Jose G and PINQUIER and Julien},
  PUBLISHER = {{ATALA \& AFPC}},
  VOLUME = {1 : articles longs et prises de position},
  PAGES = {245-261},
  YEAR = {2024},
  MONTH = Jul,
  KEYWORDS = {n{\'e}ologisme ; terminologie ; morphologie ; traduction automatique},
  PDF = {https://inria.hal.science/hal-04623021/file/9096.pdf},
  HAL_ID = {hal-04623021},
  HAL_VERSION = {v1},
}
```

```
@inproceedings{peng-etal-2024-propos,
    title = "{{\`A}} propos des difficult{\'e}s de traduire automatiquement de longs documents",
    author = "Peng, Ziqian  and
      Bawden, Rachel  and
      Yvon, Fran{\c{c}}ois",
    editor = "Balaguer, Mathieu  and
      Bendahman, Nihed  and
      Ho-dac, Lydia-Mai  and
      Mauclair, Julie  and
      G Moreno, Jose  and
      Pinquier, Julien",
    booktitle = "Actes de la 31{\`e}me Conf{\'e}rence sur le Traitement Automatique des Langues Naturelles, volume 1 : articles longs et prises de position",
    month = "7",
    year = "2024",
    address = "Toulouse, France",
    publisher = "ATALA and AFPC",
    url = "https://aclanthology.org/2024.jeptalnrecital-taln.1/",
    pages = "2--21",
    language = "fra",
}
```

<!-- https://github.com/PaulLerner/neott -->

<!-- Pytorch 1.13.1
transformers 4.45.23 -->
