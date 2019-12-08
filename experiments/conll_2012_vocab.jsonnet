{
    "vocabulary": {
        "type": "extended",
        "max_vocab_size": {"tokens": 10000}
    },
    "datasets_for_vocab_creation": ["train"],
    "dataset_reader": {
        "type": "conll2012_jsonl",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false
            }
        }
    },
    "train_data_path": "data/conll-2012/processed/train.jsonl",
}
