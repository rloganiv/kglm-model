{
    "vocabulary": {
        "type": "extended",
        "max_vocab_size": {
            // This does not count the @@UNKNOWN@@ token, which
            // ends up being our 10,000th token.
            "tokens": 9999
        }
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
    "train_data_path": "data/conll-2012/processed/train.jsonl",
    "validation_data_path": "data/conll-2012/processed/dev.jsonl",
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
        "max_mention_length": 100,
        "max_embeddings": 100,
        "tie_weights": true,
        "dropout_rate": 0.4,
        "variational_dropout_rate": 0.1
    },
    "iterator": {
        "type": "fancy",
        "batch_size": 16,
        "split_size": 30,
        "splitting_keys": [
            "source",
            "entity_types",
            "entity_ids",
            "mention_lengths"
        ],
    },
    "trainer": {
        "type": "lm",
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3
        }
    }
}
