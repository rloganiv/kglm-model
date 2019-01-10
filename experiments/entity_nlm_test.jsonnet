{
    "dataset_reader": {
        "type": "enhanced-wikitext",
        "enumerate_entities": true,
    },
    "train_data_path": "kglm/tests/fixtures/mini.train.jsonl",
    "validation_data_path": "kglm/tests/fixtures/mini.train.jsonl",
    "model": {
        "type": "entitynlm",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                },
            },
        },
        "dim": 10,
        "max_length": 10
    },
    "iterator": {
        "type": "split",
        "splitter": {
            "type": "random",
            "mean_split_size": 4,
            "min_split_size": 4,
            "max_split_size": 10,
            "splitting_keys": ["input", "output", "z", "e", "l"],
        },
        "sorting_keys": [["input", "num_tokens"]],
    },
    "trainer": {
        "num_epochs": 40,
        "patience": 5,
        "grad_norm": 5.0,
        "optimizer": {
        "type": "adam",
        "lr": 0.001
        }
    }
}