import pandas as pd
import os

from spacy.util import ensure_path
from spacy.language import Language
from spacy.tokens import Doc
import srsly
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

@Language.factory("nclf")
class NeighborhoodClf:
    def __init__(self, nlp, name: str):
        self.name = name
        self.vocab = nlp.vocab
        self.block_labels = None
        self.neighborhood_labels = None

    def initialize(self, get_examples=None, nlp=None, blocks_path:str=None, neighborhood_path:str=None):
        self.block_labels = pd.read_parquet(blocks_path).groupby('block_name')['community_name'].agg(list).to_dict()
        self.neighborhood_labels = pd.read_parquet(neighborhood_path).groupby('name')['community_name'].agg(list).to_dict()

    def to_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        if not path.exists():
            path.mkdir()
        self.block_labels.to_json(path / "block_labels.json")
        self.neighborhood_labels.to_json(path / "neighborhood_labels.json")

    def from_disk(self, path, exclude=tuple()):
        self.block_labels = srsly.read_json(path / "block_labels.json")
        self.neighborhood_labels = srsly.read_json(path / "neighborhood_labels.json")
        return self
    
    def __call__(self, doc: Doc):
        for label in self.block_labels.get(doc.text, []):
            if not hasattr(doc._, 'block_cats') or doc._.block_cats is None:
                doc._.block_cats = {}
            doc._.block_cats[label] = 1
        for label in self.neighborhood_labels.get(doc.text, []):
            if not hasattr(doc._, 'neighborhood_cats') or doc._.block_cats is None:
                doc._.block_cats = {}
            doc._.block_cats[label] = 1
        return doc