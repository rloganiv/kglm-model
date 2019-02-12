{
    "dataset_reader": {
        "type": "language_modeling",
        "tokens_per_instance": 80,
        "tokenizer": {
            "type": "word",
            "word_splitter": {"type": "just_spaces"},
        },
    },
    "train_data_path": "../awd-lstm-lm/data/enhanced-wikitext/train.txt",
    "validation_data_path": "../awd-lstm-lm/data/enhanced-wikitext/valid.txt",
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
        "type": "basic",
        "batch_size": 60,
    },
    "trainer": {
        "num_epochs": 500,
        "cuda_device": 1,
        "grad_clipping": 0.25,
        "optimizer": {
            "type": "sgd",
            "lr": 30,
            "weight_decay": 1.2e-6
        },
        "patience": 10,
        "validation_metric": "-ppl"
    }
}
