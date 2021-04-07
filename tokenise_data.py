"""Function for tokenising NER dataset."""
import os

from datasets.dataset_dict import DatasetDict


def tokenise_data(tokeniser, data, uncased=False):
    """Adds 'input_ids' field to dataset.

    Args:
        tokeniser: tokeniser instance
        data: Dataset or DatasetDict with 'text' index containing strings, and
        'label' index containing integers
        uncased: converts all text to lowercase before tokenising

    Returns:
        Same container as data with additional fields 'input_ids' and optional
        'attention_mask'.

    Raises:
        KeyError if 'text' or 'label' field is not present in dataset.
    """
    def tokenise(examples):
        if uncased:
            text = [eg.lower() for eg in examples['text']]
        else:
            text = examples['text']
        examples['label']  # raises KeyError if 'label' field is missing
        return tokeniser(text)
    # Number of processes is half of CPUs or size of smallest (sub)set
    if isinstance(data, DatasetDict):
        min_samples = min([len(split) for split in data])
    else:
        min_samples = len(data)
    num_proc = min(os.cpu_count() // 2, min_samples)
    return data.map(tokenise, batched=True, num_proc=num_proc)
