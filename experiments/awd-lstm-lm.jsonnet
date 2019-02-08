{
    "vocabulary": {
        "type": "extended",
        "min_count": {"tokens": 6} // Really 3, but now we're double counting
    },
    "dataset_reader": {
        "type": "enhanced-wikitext"
    },
    "train_data_path": "./data/mini.train.jsonl",
    "validation_data_path": "./data/final.valid.jsonl",
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "awd-lstm-lm",
        "embedding_size": 400,
        "hidden_size": 1150,
        "num_layers": 3,
        "tie_weights": true,
        // "alpha": 0.0,
        // "beta": 0.0,
        "dropouth": 0.2,
        "initializer": [
            ["embedder.weight", {"type": "uniform", "a": -0.1, "b": 0.1}],
            ["decoder.bias", {"type": "constant", "val": 0.0}]

        ]
    },
    "iterator": {
        "type": "awd",
        "batch_size": 80,
        "split_size": 70,
        // "splitting_keys": ["source", "target"]
    },
    // "validation_iterator": {
    //     "type": "fancy",
    //     "batch_size": 10,
    //     "split_size": 70,
    //     "splitting_keys": ["source", "target"]
    // },
    "trainer": {
        "num_epochs": 500,
        "cuda_device": 0,
        "grad_clipping": 0.25,
        "optimizer": {
            "type": "sgd",
            "lr": 30.1,
            "weight_decay": 1.2e-6
        },
        "patience": 10,
        "validation_metric": "-ppl"
    }
}
