import pandas as pd
import os

from spacy.util import ensure_path
from spacy.language import Language
from spacy.tokens import Doc
import srsly
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

# XXX: TODO: WE DONT NEED THIS TO BE A COMPONENT!
#           ACTUALLY WE'LL JUST USE AN OTS TEXTCAT
#           BECAUSE WE WANT THE MODEL TO BE FLEXIBLE
#           IT WILL ACTUALLY JUST TAKE PLACE ENTITIES, 
#           NOT WHOLE SENTENCES!
#           BUT WE WILL JUST USE THE GEOMS TO GENERATE POSITIVE
#           TRAINING EXAMPLES! THEN THE MODEL WILL BASICALLY
#           COMPRESS THAT INFORMATION VIA EMBEDDINGS
#           INTEAD OF MEMORIZING EVERY VARIATION
@Language.factory("nclf")
class NeighborhoodClf:
    def __init__(self, nlp, name: str):
        logger.debug("calling NCLF constructor...")
        self.name = name
        self.vocab = nlp.vocab
        self.labels = []

    def initialize(self, get_examples=None, nlp=None, labels_path:str=None):
        logger.debug("calling NCLF intialize...")
        logger.debug("found labels path? {}".format(os.path.exists(labels_path)))
        df = pd.read_parquet(labels_path, columns=['community_name'])
        logger.debug("found {} communities".format(len(df)))
        self.labels = df['community_name'].tolist()

    def to_disk(self, path, exclude=tuple()):
        path = ensure_path(path)
        if not path.exists():
            path.mkdir()
        data_path = path / "labels.json"
        logger.debug("writing {} communities".format(len(self.labels)))
        srsly.write_json(data_path, self.labels)

    def from_disk(self, path, exclude=tuple()):
        data_path = path / "labels.json"
        self.labels = srsly.read_json(data_path)
        logger.debug("loaded {} communities".format(len(self.labels)))
        logger.debug("loaded {}".format(self.labels))
        return self
    
    def __call__(self, doc: Doc):
        return doc