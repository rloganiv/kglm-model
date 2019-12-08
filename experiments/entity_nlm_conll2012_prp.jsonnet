{
    "vocabulary": {
        "type": "extended",
        "directory_path": "/kermit/rlogan/entity-nlm/data/vocabulary",
        "extend": false
    },
    "dataset_reader": {
        "type": "conll2012_jsonl",
    },
    "train_data_path": "/kermit/rlogan/entity-nlm/data/conll-2012/processed/train.jsonl",
    "validation_data_path": "/kermit/rlogan/entity-nlm/data/conll-2012/processed/dev.jsonl",
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "entitynlm",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 200,
                    "trainable": true
                },
            },
        },
        "embedding_dim": 200,
        "hidden_size": 200,
        "num_layers": 1,
        "max_mention_length": 100,
        "max_embeddings": 100,
        "tie_weights": false,
        "dropout_rate": 0.2,
        "variational_dropout_rate": 0.2
    },
    "iterator": {
        "type": "fancy",
        "batch_size": 60,
        "split_size": 70,
        "splitting_keys": [
            "source",
            "entity_types",
            "entity_ids",
            "mention_lengths"
        ],
    },
    "validation_iterator": {
        "type": "fancy",
        "batch_size": 60,
        "split_size": 70,
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
            "lr": 3e-4,
        }
    }
}
