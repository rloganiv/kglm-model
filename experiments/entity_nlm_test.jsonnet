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
                    "embedding_dim": 300,
                    "trainable": true
                },
            },
        },
    },
    "trainer": {
        "num_epochs": 40,
        "patience": 5,
        "grad_norm": 5.0,
        "validation_metric": "+accuracy",
        "optimizer": {
        "type": "adam",
        "lr": 0.001
        }
    }
}