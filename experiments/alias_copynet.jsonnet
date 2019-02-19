{
    "vocabulary": {
        "type": "extended",
        "extend": false,
        "directory_path": "data/enhanced-wikitext-2-vocab"
    },
    "dataset_reader": {
        "type": "enhanced-wikitext-kglm",
        "alias_database_path": "data/enhanced-wikitext-2.alias.pkl"
    },
    "train_data_path": "data/enhanced-wikitext-2.train.jsonl",
    "validation_data_path": "data/enhanced-wikitext.valid.jsonl",
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "alias-copynet",
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
                    "pretrained_file": "data/enhanced-wikitext-2.embeddings.txt",
                    "embedding_dim": 400,
                    "trainable": false,
                    "vocab_namespace": "entity_ids"
                }
            }
        },
        "alias_encoder": {
            "type": "lstm",
            "input_size": 400,
            "hidden_size": 400
        },
        "hidden_size": 1150,
        "num_layers": 3,
        "tie_weights": true,
        "dropouth": 0.2,
        "initializer": [
            ["token_embedder.weight", {"type": "uniform", "a": -0.1, "b": 0.1}],
            ["decoder.bias", {"type": "constant", "val": 0.0}]
        ]
    },
    "iterator": {
        "type": "fancy",
        "batch_size": 30,
        "split_size": 70,
        "splitting_keys": [
                "source",
                "target",
                "entity_ids",
                "shortlist_inds",
                "alias_copy_inds"
        ]
    },
    "validation_iterator": {
        "type": "fancy",
        "batch_size": 10,
        "split_size": 70,
        "splitting_keys": [
                "source",
                "target",
                "entity_ids",
                "shortlist_inds",
                "alias_copy_inds",
        ]
    },
    "trainer": {
        "type": "lm",
        "num_epochs": 500,
        "cuda_device": 0,
        "grad_clipping": 0.025,
        "optimizer": {
            "type": "nt-asgd",
            "lr": 30,
            "weight_decay": 1.2e-6
        },
        "learning_rate_scheduler": {
            "type": "nt-asgd",
            "non_monotone_interval": 5
        },
        "validation_metric": "-ppl"
    }
}
