import pandas as pd
import geopandas as gpd

from spacy.tokens import Doc, Span, DocBin
from spacy.util import filter_spans, ensure_path
from spacy.language import Language
from spacy.matcher.phrasematcher import PhraseMatcher
from spacy.matcher.matcher import Matcher

from scripts.geoms.operations import sides

class MatcherBase:
    def __init__(self, nlp, name: str, overwrite: bool=False):
        self.name = name
        self.matcher = None
        self.patterns = None
        self.vocab = nlp.vocab
        self.overwrite = overwrite

    def to_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        path = ensure_path(path)
        if not path.exists():
            path.mkdir()
        data_path = path / "patterns.spacy"
        db = DocBin()
        for d in self.patterns:
            db.add(d)
        db.to_disk(data_path)

    def from_disk(self, path, exclude=tuple()):
        # This will receive the directory path + /my_component
        data_path = path / "patterns.spacy"
        self.patterns = list(DocBin().from_disk(data_path).get_docs(self.vocab))
        return self
    
    def _skip(self, doc: Doc, match: Span):
        return any(t.ent_type for t in doc[match.start:match.end]) and not self.overwrite

    def __call__(self, doc: Doc):
        matches = self.matcher(doc, as_spans=True)
        matches = [m for m in matches if not self._skip(doc, m)]
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc
    

@Language.factory("gpe_matcher")
class GPEMatcher(MatcherBase):
    def from_disk(self, path, exclude=tuple()):
        super().from_disk(path, exclude)
        self.matcher = PhraseMatcher(self.vocab)
        self.matcher.add("GPE", self.patterns)
        return self

    def initialize(self, get_examples=None, nlp=None, comm_area_path: str=None, neighborhood_path: str=None):
        gpes = pd.concat([gpd.read_parquet(comm_area_path)['community_name'].rename('name'),
                            pd.read_csv(neighborhood_path)['name'],
                            pd.Series(sides)], ignore_index=True)
        gpes = gpes.str.split(",", expand=False).explode()
        gpes = gpes.str.title().drop_duplicates().sort_values()
        
        self.matcher = PhraseMatcher(self.vocab)
        self.patterns = list(nlp.tokenizer.pipe(gpes))
        self.matcher.add("GPE", self.patterns)    
    
@Language.factory("street_matcher")
class StreetMatcher(MatcherBase):
    def from_disk(self, path, exclude=tuple()):
        super().from_disk(path, exclude)
        self.matcher = PhraseMatcher(self.vocab)
        self.matcher.add("FAC", self.patterns)
        return self

    def initialize(self, get_examples=None, nlp=None, street_name_path: str=None):
        street_names = pd.read_csv(street_name_path)
        street_names = street_names.filter(like='combined').melt()['value']
        street_names = street_names.str.title().drop_duplicates().sort_values()

        self.matcher = PhraseMatcher(self.vocab)
        self.patterns = list(nlp.tokenizer.pipe(street_names))
        self.matcher.add("FAC", self.patterns)

@Language.component("street_to_neighborhood")
def street_to_neighborhood(doc: Doc):
    # Fix street names that are actually neighborhoods
    new_ents = []
    for idx, ent in enumerate(doc.ents):
        if ent.label_ == "FAC" and (ent.end != len(doc.ents)):
            next_token = doc[ent.end]
            # Must match [FAC] neighborhood
            if next_token.text == "neighborhood":
                new_ent = Span(doc, ent.start, ent.end, label="GPE")
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc.ents = new_ents
    return doc

@Language.factory("age_matcher")
class AgeMatcher:
    def __init__(self, nlp, name: str, overwrite: bool=False):
        self.name = name
        self.matcher = Matcher(nlp.vocab)
        self.pattern = {"TEXT": {"REGEX": r"\d+[ -]year[ -]old"}}
        self.matcher.add("CARDINAL", [[self.pattern]])
        self.overwrite = overwrite

    def _skip(self, doc: Doc, match: Span):
        return any(t.ent_type for t in doc[match.start:match.end]) and not self.overwrite

    def __call__(self, doc: Doc):
        matches = self.matcher(doc, as_spans=True)
        matches = [m for m in matches if not self._skip(doc, m)]
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc

@Language.component("block_matcher")
def block_matcher(doc: Doc):
    new_ents = []
    for idx, ent in enumerate(doc.ents):
        # Only check for title if it's a person and not the first token
        if ent.label_ == "FAC" and ent.start >= 3 and idx >= 1:
            prev_ent = list(doc.ents)[idx-1]
            prev_tokens = doc[ent.start - 3: ent.start]
            # Must match [CARDINAL] block of [FAC]
            if (prev_tokens[2].text == "of" and prev_tokens[1].text == "block"
                and prev_ent.label_ == "CARDINAL" and prev_tokens[0].text == prev_ent.text):
                new_ent = Span(doc, ent.start - 3, ent.end, label=ent.label)
                new_ents.append(new_ent)
    doc.ents = filter_spans(list(doc.ents) + new_ents)
    return doc

@Language.component("intersection_matcher")
def intersection_matcher(doc: Doc):
    new_ents = []
    for idx, ent in enumerate(doc.ents):
        # Only check for title if it's a person and not the first token
        if ent.label_ == "FAC" and ent.start >= 2 and idx >= 1:
            prev_ent = list(doc.ents)[idx-1]
            prev_tokens = doc[ent.start - 2: ent.start]
            # Must match [STREET] and [STREET]
            if ((prev_tokens[1].text == "and" or prev_tokens[1].text == "&")
                and prev_ent.label_ == "FAC" and prev_tokens[0].text == prev_ent.text):
                new_ent = Span(doc, ent.start - 2, ent.end, label=ent.label)
                new_ents.append(new_ent)
    doc.ents = filter_spans(list(doc.ents) + new_ents)
    return doc