import torch

def generate_batches(input: torch.Tensor):
    """
    HIGH LEVEL OVERVIEW: Break a long sequence into short sequences,then yield them one at a time.

    ADDITIONAL STUFF:
        Make sure all sequences are the same length!
        Should support AllenNLP-like token/mask dicts.
        Should support variable length backprop sequences.
    """
    raise NotImplementedError
