{
    "vocabulary": {
        "type": "extended",
        "max_vocab_size": {"tokens": 33278},
        "min_count": {"tokens": 3},
    },
    "dataset_reader": {
        "type": "enhanced-wikitext",
        "enumerate_entities": true,
    },
    "train_data_path": "data/mini.train.jsonl",
    "validation_data_path": "data/final.valid.jsonl",
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
        "encoder": {
            "type": "lstm",
            "input_size": 256,
            "hidden_size": 256,
            "dropout": 0.5,
            "stateful": true
        },
        "embedding_dim": 256,
        "max_mention_length": 166,
        "max_embeddings": 1345,
        "tie_weights": true,
        "dropout_rate": 0.4,
        "variational_dropout_rate": 0.1
    },
    "iterator": {
        "type": "split",
        "batch_size": 40,
        "splitter": {
            "type": "random",
            "mean_split_size": 70,
            "min_split_size": 30,
            "max_split_size": 100,
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
        "num_epochs": 500,
        "cuda_device": 1,
        "optimizer": {
            "type": "adam",
            "lr": 1e-3
        },
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "mode": "max",
            "min_lr": 1e-6
        },
        "patience": 20,
        "validation_metric": "-ppl",
        "should_log_learning_rate": true
    }
}
