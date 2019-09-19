{
    "vocabulary": {
        "type": "extended",
        "extend": false,
        "directory_path": "data/linked-wikitext-2/vocab"
    },
    "dataset_reader": {
        "type": "enhanced-wikitext-kglm",
        "alias_database_path": "data/linked-wikitext-2/alias.pkl",
        "mode": "discriminative"
    },
    "train_data_path": "data/linked-wikitext-2/train.jsonl",
    "validation_data_path": "data/linked-wikitext-2/valid.jsonl",
    "model": {
        "type": "kglm-disc",
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
                    "pretrained_file": "data/linked-wikitext-2/embeddings.entities.txt",
                    "embedding_dim": 256,
                    "trainable": false,
                    "vocab_namespace": "entity_ids"
                }
            }
        },
        "relation_embedder": {
            "token_embedders": {
                "relations": {
                    "type": "embedding",
                    "pretrained_file": "data/linked-wikitext-2/embeddings.relations.txt",
                    "embedding_dim": 256,
                    "trainable": true,
                    "vocab_namespace": "relations"
                }
            }
        },
        "knowledge_graph_path": "data/linked-wikitext-2/knowledge_graph.pkl",
        "use_shortlist": false,
        "hidden_size": 1150,
        "num_layers": 3,
        "cutoff": 30,
        "tie_weights": true,
        "initializer": [
            ["token_embedder.weight", {"type": "uniform", "a": -0.1, "b": 0.1}],
            ["decoder.bias", {"type": "constant", "val": 0.0}]
        ]
    },
    "iterator": {
        "type": "fancy",
        "batch_size": 60,
        "split_size": 70,
        "splitting_keys": [
                "source",
                "mention_type",
                "raw_entity_ids",
                "entity_ids",
                "parent_ids",
                "relations",
                "shortlist_inds"
        ]
    },
    "validation_iterator": {
        "type": "fancy",
        "batch_size": 60,
        "split_size": 70,
        "splitting_keys": [
                "source",
                "mention_type",
                "raw_entity_ids",
                "entity_ids",
                "parent_ids",
                "relations",
                "shortlist_inds"
        ],
        "truncate": false
    },
    "trainer": {
        "type": "lm",
        "num_epochs": 500,
        "cuda_device": 0,
        // "grad_clipping": 0.25,
        // "optimizer": {
        //     "type": "nt-asgd",
        //     "lr": 22.5,
        //     "weight_decay": 1.2e-6
        // },
        // "learning_rate_scheduler": {
        //     "type": "nt-asgd",
        //     "non_monotone_interval": 5
        // },
        "optimizer": {
            "type": "adam",
            "lr": 3e-4,
            "weight_decay": 1.2e-6
        },
    }
}
