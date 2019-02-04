{
    "vocabulary": {
        "type": "extended",
        "min_count": {"tokens": 3}
    },
    "dataset_reader": {
        "type": "enhanced-wikitext"
    },
    "train_data_path": "./data/mini.train.jsonl",
    "validation_data_path": "./data/final.valid.jsonl",
    "model": {
        "type": "awd-lstm-lm",
        "embedding_size": 400,
        "hidden_size": 1150,
        "num_layers": 3,
        "tie_weights": true,
        "dropouth": 0.2,
        "initializer": [
            ["embedder.weight", {"type": "uniform", "a": -0.1, "b": 0.1}]
        ]
    },
    "iterator": {
        "type": "split",
        "batch_size": 80,
        "splitter": {
            "type": "random",
            "mean_split_size": 70,
            "max_split_size": 80,
            "min_split_size": 60,
            "splitting_keys": ["tokens"]
        },
        "sorting_keys": [["tokens", "num_tokens"]]
    },
    "trainer": {
        "num_epochs": 500,
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 3e-4,
        },
        "patience": 10,
        "validation_metric": "-ppl"
    }
}
