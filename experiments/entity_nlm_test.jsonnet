{
    "vocabulary": {
        "type": "extended",
        "max_vocab_size": {"tokens": 50000},
    },
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
                    "embedding_dim": 100,
                    "trainable": true
                },
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 100,
            "stateful": true
        },
        "embedding_dim": 100,
        "max_mention_length": 100,
        "max_embeddings": 1000,
        "tie_weights": true,
    },
    "iterator": {
        "type": "split",
        "batch_size": 8,
        "splitter": {
            "type": "random",
            "mean_split_size": 60,
            "min_split_size": 30,
            "max_split_size": 70,
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
            "lr": 3e-4
        }
    }
}