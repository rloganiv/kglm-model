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
For convenience, we provide pre-trained embeddings and pickled dictionaries containing the relevant portions of Wikidata [here](https://drive.google.com/file/d/1tvcbUY9CUQ770igxG9pQZA4qjmwFb5Cs/view?usp=sharing).


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
    [input_file]
```
where:
- `model_archive_file` - Trained (generative) model checkpoint. This is the model whose perplexity will be evaluated.
- `sampler_archive_file` - Trained (discriminative) model checkpoint. This is the model used to create annotations during importance sampling. See Section 4 of the paper for more details about importance sampling.
- `input_file` - Dataset to measure perplexity on.


Sentence Completion
---
To perform sentence completion experiments run:
```
allennlp predict --predictor cloze [model_archive_file] [input_file]
```
where:
- `model_archive_file` - Trained (generative) model checkpoint.
- `input_file` - Sentence completion dataset.

Input data is expected to be in JSON lines format, where each object has the following fields:
- `prefix` - The sentence to complete.
- `entity_id` - (Optional) ID of the parent entity.
- `nel` - (Optional) Output of an named entity linker on the prefix.
- `shortlist` - (Optional) Restricted list of possible parent IDs.