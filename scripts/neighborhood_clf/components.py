import json
import pandas as pd
from itertools import chain
from scripts.utils.logging import setup_logger

from spacy.language import Language
from spacy.tokens import Doc
from spacy.util import ensure_path
from spacy.vocab import Vocab
from string import punctuation
import srsly

logger = setup_logger(__name__)


@Language.factory("nclf", assigns=["doc._.cats"])
def make_nclf(
    nlp: Language,
    name: str,
) -> "NeighborhoodClf":
    """Create a NeighborhoodClf component. The text categorizer predicts categories
    over a whole document. It can learn one or more labels, and the labels are considered
    to be inclusive (i.e. multiple true labels per doc).
    """
    return NeighborhoodClf(nlp.vocab, name)

class NeighborhoodClf:
    Doc.set_extension("cats", default={}, force=True)

    def __init__(
        self, 
        vocab: Vocab, 
        name: str, 
    )-> None:
        self.vocab = vocab
        self.name = name
        # self.labels = []
        self.block_labels = None
        self.neighborhood_labels = None

    def initialize(self, get_examples=None, nlp=None, blocks_path:str=None, neighborhood_path:str=None):
        self.block_labels = (pd.read_parquet(blocks_path)
                             .assign(block_name = lambda x: x['block_name'].str.upper())
                             .groupby('block_name')
                             ['community_name'].unique())
        self.neighborhood_labels = (pd.read_csv(neighborhood_path)
                                    .assign(name = lambda x: x['name'].str.title())
                                    .groupby('name')
                                    ['community_name'].unique())
        # self.labels = set(chain.from_iterable(self.block_labels.values())) 
        # self.labels |= set(chain.from_iterable(self.neighborhood_labels.values()))
        # self.labels = list(self.labels)

    def to_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        if not path.exists():
            path.mkdir()
        # TODO: Figuring out this serialization for evaluation. But need to re-run training to write to disk properly.
        pd.Series(self.block_labels,name='community_names') \
         .to_frame() \
         .to_parquet(path / "block_labels.parquet")
        pd.Series(self.neighborhood_labels,name='community_names') \
         .to_frame() \
         .to_parquet(path / "neighborhood_labels.parquet")
        
    def from_disk(self, path, exclude=tuple()):
        self.block_labels = pd.read_parquet(path / "block_labels.parquet")['community_names']
        self.neighborhood_labels = pd.read_parquet(path / "neighborhood_labels.parquet")['community_names']
        return self
    
    def __call__(self, doc: Doc):
        gpe_shape = doc.user_data.get("gpe_shape", None)
        if gpe_shape == "block":
            for label in self.block_labels.get(doc.text, []):
                logger.debug("Matched gpe block: %s --> %s", doc.text, label)
                doc._.cats[label] = 1
        elif gpe_shape == "neighborhood":
            for label in self.neighborhood_labels.get(doc.text, []):
                logger.debug("Matched gpe neighborhood: %s --> %s", doc.text, label)
                doc._.cats[label] = 1
        elif pd.notna(gpe_shape):
            logger.warning("Cant match gpe %s", gpe_shape)
        else:
            for label in self.block_labels.get(doc.text, []):
                logger.debug("Matched block: %s --> %s", doc.text, label)
                doc._.cats[label] = 1
            for label in self.neighborhood_labels.get(doc.text, []):
                logger.debug("Matched neighborhood: %s --> %s", doc.text, label)
                doc._.cats[label] = 1
        return doc
    
@Language.component("cat_merger")
def cat_merger(doc: Doc) -> Doc:
    cats = doc.cats.keys() | doc._.cats.keys()
    doc.cats = {c: max(doc.cats.get(c, 0), doc._.cats.get(c, 0)) for c in cats}
    doc._.cats = None
    return doc