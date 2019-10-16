{
    "vocabulary": {
        "type": "extended",
        "extend": false,
        "directory_path": "data/vocabulary-mini"
    },
    "dataset_reader": {
        "type": "conll2012_jsonl",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "data/conll-2012/processed/train-mini.jsonl",
    "validation_data_path": "data/conll-2012/processed/dev-mini.jsonl",
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "entitynlm",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 256,
                    "trainable": true
                },
            },
        },
        "embedding_dim": 256,
        "hidden_size": 256,
        "num_layers": 1,
        "max_mention_length": 50,
        "max_embeddings": 10,
        "tie_weights": true,
        "dropout_rate": 0.4,
        "variational_dropout_rate": 0.1
    },
    "iterator": {
        "type": "fancy",
        "batch_size": 16,
        "split_size": 15,
        "splitting_keys": [
            "source",
            "entity_types",
            "entity_ids",
            "mention_lengths"
        ],
    },
    "validation_iterator": {
        "type": "fancy",
        "batch_size": 16,
        "split_size": 15,
        "splitting_keys": [
            "source",
            "entity_types",
            "entity_ids",
            "mention_lengths"
        ],
        "truncate": false
    },
    "trainer": {
        "type": "lm",
        "num_epochs": 400,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3
        }
    }
}
