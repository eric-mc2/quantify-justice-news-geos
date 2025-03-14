# TODO: Training is slow-ish. Maybe I can create a custom batch function to sort by length.

import spacy
from typing import List, Callable
from spacy.training import Example
import numpy as np

# Register this as a custom batcher
@spacy.registry.batchers("length_sorted_batches.v1")
def create_length_sorted_batcher(size: int) -> Callable[[List[Example]], List[List[Example]]]:
    def length_sorted_batcher(docs: List[Example]) -> List[List[Example]]:
        """
        This might not exactly see the whole dataset at a time. 
        Still might be worth sorting though.
        
        Note: Even though this returns a flattened list, spacy later on will
        slice it into mini-batches. Since we can't control that, then
        I won't implement the part about creating mini-1-batches for 
        extra long outlier texts. 
        """
        docs.sort(key=lambda d: len(d.reference.text))
        batches = np.array_split(docs, len(docs) // size)
        for b in batches:
            np.random.shuffle(b)
        np.random.shuffle(batches)
        return batches
    return length_sorted_batcher