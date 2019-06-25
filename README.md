Knowledge Graph Language Model
===

This repo contains an implementation of the KGLM model described in "Barack's Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling", Robert L. Logan IV, Nelson F. Liu, Matthew E. Peters, Matt Gardner and Sameer Singh, ACL 2019 [[arXiv]](https://arxiv.org/abs/1906.07241).


Setup
---
You will need Python 3.5+. Dependencies can be installed by running:
```{bash}
pip install -r requirements.txt
```

Data
---
KGLM is trained on the *Linked WikiText-2* dataset which can be downloaded at https://rloganiv.github.io/linked-wikitext-2.

Additionally, you will need embeddings for entities/relations in the [Wikidata](https://www.wikidata.org/) knowledge graph, as well as access to the knowledge graph itself (in order to look up entity aliases/related entities).
For convenience, we provide pre-trained embeddings and pickled dictionaries containing the relevant portions of Wikidata [here]().


Training
---
To train the model run:
```{bash}
allennlp train [path to config] -s [path to save checkpoint to] --include-package kglm
```
example model configurations are provided in the `experiments` directory.


Perplexity Evaluation
---
To estimate perplexity of a trained model on held-out data run:
```{bash}
python -m kglm.run evaluate-perplexity \
    [model_archive_file] \
    [sampler_archive_file] \
    [input_data]
```
where:
- `model_archive_file` - Trained (generative) model checkpoint. This is the model whose perplexity will be evaluated.
- `sampler_archive_file` - Trained (discriminative) model checkpoint. This is the model used to create annotations during importance sampling. See Section 4 of the paper for more details about importance sampling.
- `input_data` - Path to dataset to measure perplexity on.

Sentence Completion
---
To perform sentence completion experiments run:
```
allennlp predict --predictor cloze [model_archive_file] [input_file]
```
where
- `model_archive_file` - Trained (generative) model checkpoint. This is the model whose perplexity will be evaluated.
- `input_data` - Path to dataset to measure perplexity on.