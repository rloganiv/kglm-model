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
                    "embedding_dim": 512,
                    "trainable": true
                },
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": 512,
            "hidden_size": 512,
            "stateful": true
        },
        "embedding_dim": 512,
        "max_mention_length": 512,
        "initializer": [
            ["_entity_type_embeddings", {"type":  "orthogonal"}],
            ["_null_entity_embedding", {"type": "orthogonal"}],
        ],
    },
    "iterator": {
        "type": "split",
        "batch_size": 16,
        "splitter": {
            "type": "random",
            "mean_split_size": 50,
            "min_split_size": 60,
            "max_split_size": 30,
            "splitting_keys": [
                "inputs",
                "outputs",
                "entity_types",
                "entity_ids",
                "entity_mention_lengths"
            ],
        },
        "sorting_keys": [["inputs", "num_tokens"]],
    },
    "trainer": {
        "num_epochs": 40,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 3e-8
        }
    }
}