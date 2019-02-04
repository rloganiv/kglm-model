{
    "vocabulary": {
        "type": "extended",
        "min_count": {"tokens": 3},
        "pretrained_files": {
            "entity_ids": "data/mini.embeddings.400.txt"
        }
    },
    "dataset_reader": {
        "type": "enhanced-wikitext-kglm",
        "alias_database_path": "data/mini.alias.pkl"
    },
    "train_data_path": "data/mini.train.jsonl",
    "validation_data_path": "data/final.valid.jsonl",
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "kglm",
        "token_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 400,
                    "trainable": true
                }
            }
        },
        "entity_embedder": {
            "token_embedders": {
                "entity_ids": {
                    "type": "embedding",
                    "pretrained_file": "data/mini.embeddings.400.txt",
                    "embedding_dim": 400,
                    "trainable": false,
                    "vocab_namespace": "entity_ids"
                }
            }
        },
        "encoder": {
            "type": "lstm",
            "input_size": 400,
            "hidden_size": 400,
            "num_layers": 1,
            "stateful": true
        },
        "tie_weights": true,
        "dropout_rate": 0.6,
        "variational_dropout_rate": 0.3
    },
    "iterator": {
        "type": "split",
        "batch_size": 24,
        "splitter": {
            "type": "random",
            "mean_split_size": 70,
            "min_split_size": 60,
            "max_split_size": 80,
            "splitting_keys": [
                "tokens",
                "entity_identifiers",
                "shortlist_indices",
                "alias_copy_indices"
            ]
        },
        "sorting_keys": [["tokens", "num_tokens"]],
    },
    "trainer": {
        "cuda_device": 0,
        "num_epochs": 500,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3,
            "betas": [0.0, 0.999], // State of art of evaluation LMs
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.1,
            "patience": 3
        },
        "patience": 10
    }
}
