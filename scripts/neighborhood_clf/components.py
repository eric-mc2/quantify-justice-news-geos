import pandas as pd
from typing import Any,Callable, Dict, Iterable, List, Optional, Tuple
from itertools import chain
from scripts.utils.logging import setup_logger

import srsly
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import ensure_path, registry
from spacy.vocab import Vocab

logger = setup_logger(__name__)


@Language.factory(
    "nclf",
    assigns=["doc._.block_cats","doc._.neighborhood_cats"],
    # TODO: COPIED FROM TEXTCAT. NOT SURE IF CORRECT
    default_config={
        "scorer": {"@scorers": "spacy.nclf_scorer.v1"},
    },
    # TODO: COPIED FROM TEXTCAT. NOT SURE IF CORRECT
    default_score_weights={
        "nclf_cats_score": 1.0,
        "nclf_cats_score_desc": None,
        "nclf_cats_micro_p": None,
        "nclf_cats_micro_r": None,
        "nclf_cats_micro_f": None,
        "nclf_cats_macro_p": None,
        "nclf_cats_macro_r": None,
        "nclf_cats_macro_f": None,
        "nclf_cats_macro_auc": None,
        "nclf_cats_f_per_type": None,
    })
def make_nclf(
    nlp: Language,
    name: str,
    scorer: Optional[Callable],
) -> "NeighborhoodClf":
    """Create a NeighborhoodClf component. The text categorizer predicts categories
    over a whole document. It can learn one or more labels, and the labels are considered
    to be inclusive (i.e. multiple true labels per doc).
    """
    return NeighborhoodClf(nlp.vocab, name, scorer=scorer)


# TODO: COPEID FROM TEXTCAT. NOT SURE IF USED.
def nclf_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    block_scores = Scorer.score_cats(
        examples,
        "_.block_cats",
        multi_label=True,
        **kwargs,
    )
    neighborhood_scores = Scorer.score_cats(
        examples,
        "_.neighborhood_cats",
        multi_label=True,
        **kwargs,
    )
    left_keys = set(block_scores.keys()) - set(neighborhood_scores.keys())
    inner_keys = set(block_scores.keys()) & set(neighborhood_scores.keys())
    right_keys = set(neighborhood_scores.keys()) - set(block_scores.keys())
    left_scores = {k: block_scores[k] for k in left_keys}
    inner_scores = {k: (block_scores[k] + neighborhood_scores[k]) / 2 for k in inner_keys}
    right_scores = {k: neighborhood_scores[k] for k in right_keys}
    scores = left_scores | inner_scores | right_scores
    scores = {"nclf_" + k: v for k,v in scores.items()}
    return scores


# TODO: COPEID FROM TEXTCAT. NOT SURE IF USED.
@registry.scorers("spacy.nclf_scorer.v1")
def make_nclf_scorer():
    return nclf_score


class NeighborhoodClf:
    def __init__(
        self, 
        vocab: Vocab, 
        name: str, 
        scorer: Optional[Callable] = nclf_score
    )-> None:
        self.vocab = vocab
        self.name = name
        self.labels = []
        self.block_labels = None
        self.neighborhood_labels = None
        self.scorer = scorer

    def initialize(self, get_examples=None, nlp=None, blocks_path:str=None, neighborhood_path:str=None):
        self.block_labels = pd.read_parquet(blocks_path).groupby('block_name')['community_name'].agg(list).to_dict()
        self.neighborhood_labels = pd.read_csv(neighborhood_path).groupby('name')['community_name'].agg(list).to_dict()
        self.labels = set(chain.from_iterable(self.block_labels.values())) 
        self.labels |= set(chain.from_iterable(self.neighborhood_labels.values()))
        self.labels = list(self.labels)

    def to_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        if not path.exists():
            path.mkdir()
        srsly.write_jsonl(path / "block_labels.jsonl", self.block_labels)
        srsly.write_jsonl(path / "neighborhood_labels.jsonl", self.neighborhood_labels)

    def from_disk(self, path, exclude=tuple()):
        self.block_labels = srsly.read_jsonl(path / "block_labels.jsonl")
        self.neighborhood_labels = srsly.read_jsonl(path / "neighborhood_labels.jsonl")
        return self
    
    def __call__(self, doc: Doc):
        for label in self.block_labels.get(doc.text, []):
            if not hasattr(doc._, 'block_cats') or doc._.block_cats is None:
                doc._.block_cats = {}
            doc._.block_cats[label] = 1
        for label in self.neighborhood_labels.get(doc.text, []):
            if not hasattr(doc._, 'neighborhood_cats') or doc._.neighborhood_cats is None:
                doc._.neighborhood_cats = {}
            doc._.neighborhood_cats[label] = 1
        return doc
    