{
    "dataset_reader": {
        "type": "enhanced-wikitext-entity-nlm"
    },
    "iterator": {
        "type": "fancy",
        "batch_size": 30,
        "split_size": 70,
        "splitting_keys": [
            "source",
            "entity_types",
            "entity_ids",
            "mention_lengths"
        ]
    },
    "validation_iterator": {
        "type": "fancy",
        "batch_size": 30,
        "split_size": 70,
        "splitting_keys": [
            "source",
            "entity_types",
            "entity_ids",
            "mention_lengths"
        ],
        "truncate": false
    },
    "model": {
        "type": "entitynlm",
        "dropout_rate": 0.5,
        "embedding_dim": 400,
        "hidden_size": 1150,
        "max_embeddings": 3000,
        "max_mention_length": 100,
        "num_layers": 3,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 400,
                    "trainable": true
                }
            }
        },
        "tie_weights": true,
        "variational_dropout_rate": 0.5
    },
    "train_data_path": "data/enhanced-wikitext-2/train.jsonl",
    "validation_data_path": "data/enhanced-wikitext-2/valid.jsonl",
    "trainer": {
        "type": "lm",
        "cuda_device": 0,
        "num_epochs": 750,
        "optimizer": {
            "type": "adam",
            "lr": 0.0003
        }
    },
    "vocabulary": {
        "type": "extended",
        "directory_path": "data/enhanced-wikitext-2/vocab",
        "extend": false
    }
}
