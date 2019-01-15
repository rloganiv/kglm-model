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
        "max_mention_length": 100,
        "max_embeddings": 1000,
        "initializer": [
            ["_dummy_entity_embedding", {"type":  "xavier_uniform"}],
            ["_dummy_context_embedding", {"type": "xavier_uniform"}],
        ],
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
                "mention_lengths"
            ],
        },
        "sorting_keys": [["tokens", "num_tokens"]],
    },
    "trainer": {
        "num_epochs": 40,
        "optimizer": {
            "type": "adam",
            "lr": 3e-8
        }
    }
}