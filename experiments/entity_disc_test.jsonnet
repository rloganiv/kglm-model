{
    "vocabulary": {
        "type": "extended",
        "extend": false,
        "directory_path": "./results/entity-nlm-wt2.fixed-vocab.dropout.2/vocabulary"
    },
    "dataset_reader": {
        "type": "enhanced-wikitext",
        "enumerate_entities": true,
    },
    "train_data_path": "kglm/tests/fixtures/mini.train.jsonl",
    "validation_data_path": "kglm/tests/fixtures/mini.train.jsonl",
    "model": {
        "type": "entitydisc",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 256,
                    "trainable": true
                },
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": 256,
            "hidden_size": 256,
            "dropout": 0.5,
            "stateful": true
        },
        "embedding_dim": 256,
        "max_mention_length": 180,
        "max_embeddings": 1000,
        "dropout_rate": 0.4,
        "variational_dropout_rate": 0.1
    },
    "iterator": {
        "type": "split",
        "batch_size": 16,
        "splitter": {
            "type": "random",
            "mean_split_size": 30,
            "min_split_size": 20,
            "max_split_size": 40,
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
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3
        }
    }
}