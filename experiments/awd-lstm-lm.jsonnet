{
    "vocabulary": {
        "type": "extended",
        "min_count": {"tokens": 3}
    },
    "dataset_reader": {
        "type": "enhanced-wikitext"
    },
    "train_data_path": "data/enhanced-wikitext-2.train.jsonl",
    "validation_data_path": "data/enhanced-wikitext.valid.jsonl",
    "datasets_for_vocab_creation": ["train"],
    "model": {
        "type": "awd-lstm-lm",
        "embedding_size": 400,
        "hidden_size": 1150,
        "num_layers": 3,
        "tie_weights": true,
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
    },
    "trainer": {
        "type": "lm",
        "num_epochs": 750,
        "cuda_device": 0,
        "grad_clipping": 0.25,
        "optimizer": {
            "type": "nt-asgd",
            "lr": 30,
            "weight_decay": 1.2e-6
        },
        "learning_rate_scheduler": {
            "type": "nt-asgd",
            "non_monotone_interval": 5,
        },
        "validation_metric": "-ppl"
    }
}
