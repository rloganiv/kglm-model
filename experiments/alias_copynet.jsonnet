{
    "vocabulary": {
        "type": "extended",
        "min_count": {"tokens": 6}
    },
    "dataset_reader": {
        "type": "enhanced-wikitext-kglm",
        "alias_database_path": "data/mini.alias.pkl"
    },
    "train_data_path": "data/mini.train.jsonl",
    "validation_data_path": "data/final.valid.jsonl",
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
                    "pretrained_file": "data/mini.embeddings.400.txt",
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
        "batch_size": 32,
        "split_size": 70,
        "splitting_keys": [
                "source",
                "target",
                "entity_ids",
                "shortlist_inds",
                "alias_copy_inds"
        ]
    },
    "trainer": {
        "num_epochs": 500,
        "cuda_device": 0,
        "grad_clipping": 0.10,
        "optimizer": {
            "type": "sgd",
            "lr": 30.1,
            "weight_decay": 1.2e-6
        },
        "patience": 10,
        "validation_metric": "-ppl"
    }
}
