{
    "dataset_reader": {
        "type": "enhanced-wikitext-kglm",
        "alias_database_path": "data/mini.alias.pkl"
    },
    "train_data_path": "kglm/tests/fixtures/mini.train.jsonl",
    "validation_data_path": "kglm/tests/fixtures/mini.train.jsonl",
    "model": {
        "type": "kglm",
        "token_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "trainable": true
                },
            },
        },
        "entity_embedder": {
            "token_embedders": {
                "entity_ids": {
                    "type": "embedding",
                    "embedding_dim": 10,
                    "trainable": true
                },
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": 10,
            "hidden_size": 10,
            "stateful": true
        },
        "embedding_dim": 10,
    },
    "iterator": {
        "type": "split",
        "batch_size": 2,
        "splitter": {
            "type": "random",
            "mean_split_size": 10,
            "min_split_size": 12,
            "max_split_size": 8,
            "splitting_keys": [
                "tokens",
                "entity_types",
                "entity_ids",
                "alias_ids"
            ],
        },
        "sorting_keys": [["tokens", "num_tokens"]],
    },
    "trainer": {
        "num_epochs": 40,
        "optimizer": {
            "type": "adam",
            "lr": 3e-4
        },
    },
}