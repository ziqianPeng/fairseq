# Fine-tuning MBART Models Using BLEU for Checkpoint Selection 


This branch is adapted from the fairseq framework, for the purpose of fine-tuning mBART models using BLEU scores to select the best checkpoint for the machine translation task.

The essential modifications are in 
```
fairseq/tasks/translation_multi_simple_epoch_TranslationTask.py
```

Here is an example to run the code:

1. Preprocess the dataset, check `example_scripts/preprocess-encode.sh`
2. Train or fine-tuning the model with the prepared dataset, check `example_scripts/mBART50_FT_scipar.sh`

These two scripts are used to fine-tune `mBART50-one-to-many` on the `SciPar` corpora for the translation from English to French.
The resulting checkpoint is appiled as baseline model in the following publications:

```
Paul Lerner, François Yvon. Towards the Machine Translation of Scientific Neologisms. 2024. ⟨hal-04835653v2⟩
```

```
Paul Lerner and François Yvon. 2024. Vers la traduction automatique des néologismes scientifiques. In Actes de la 31ème Conférence sur le Traitement Automatique des Langues Naturelles, volume 1 : articles longs et prises de position, pages 245–261, Toulouse, France. ATALA and AFPC.
```

```
Ziqian Peng, Rachel Bawden, and François Yvon. 2024. À propos des difficultés de traduire automatiquement de longs documents. In Actes de la 31ème Conférence sur le Traitement Automatique des Langues Naturelles, volume 1 : articles longs et prises de position, pages 2–21, Toulouse, France. ATALA and AFPC.
```

<!-- https://github.com/PaulLerner/neott -->

<!-- Pytorch 1.13.1
transformers 4.45.23 -->
