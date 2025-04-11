import pandas as pd
import geopandas as gpd
from enum import Flag, auto
import re

import spacy
from spacy.tokens import Doc, Span, DocBin
from spacy.util import filter_spans, ensure_path
from spacy.language import Language
from spacy.matcher.phrasematcher import PhraseMatcher
from spacy.matcher.matcher import Matcher
import srsly

from scripts.geoms.operations import sides
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

class MatcherBase:
    Span.set_extension("gpe_shape", default="", force=True)
    Doc.set_extension("ents", default=[], force=True)

    def __init__(self, nlp, name: str, overwrite: bool=False):
        # Stateless
        self.name = name
        self.overwrite = overwrite
        # Must initialize
        self.vocab = nlp.vocab
        # Must to_disk / from_disk
        self.matcher = None
        self.patterns = None
        self.gpe_shape = None

    def to_disk(self, path, exclude=tuple()):
        if self.patterns is None:
            logger.warning("Nothing to serialize for %s", self.__class__.__name__)
            return
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
    
    def _skip(self, match: Span):
        return any(compare(match, s) in Comparison.OVERLAP for s in match.doc._.ents) and not self.overwrite
        
    def __call__(self, doc: Doc):
        matches = self.matcher(doc, as_spans=True)
        matches = [m for m in matches if not self._skip(m)]
        doc = apply_ents(doc, matches, self.__class__.__name__, self.gpe_shape)
        return doc
    

class GPEMatcher(MatcherBase):
    def from_disk(self, path, exclude=tuple()):
        super().from_disk(path, exclude)
        self.matcher = PhraseMatcher(self.vocab)
        self.matcher.add("GPE", self.patterns)
        return self

    def initialize(self, get_examples=None, nlp=None, gpes:pd.Series=None):
        gpes = gpes.str.split(",", expand=False).explode()
        gpes = gpes.str.title().drop_duplicates().sort_values()
        
        self.matcher = PhraseMatcher(self.vocab)
        self.patterns = list(nlp.tokenizer.pipe(gpes))
        self.matcher.add("GPE", self.patterns)    


@Language.factory("neighborhood_matcher")
class HoodMatcher(GPEMatcher):
    def __init__(self, nlp, name: str, overwrite: bool=False):
        super().__init__(nlp, name, overwrite)
        self.gpe_shape = "neighborhood"

    def initialize(self, get_examples=None, nlp=None, neighborhood_path: str=None):
        gpes = pd.read_parquet(neighborhood_path)['name'] + " neighborhood"
        super().initialize(get_examples, nlp, gpes)


@Language.factory("street_matcher")
class StreetMatcher(MatcherBase):
    def __init__(self, nlp, name: str, overwrite: bool=False):
        super().__init__(nlp, name, overwrite)
        self.gpe_shape = "street"
    
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


@Language.factory("community_matcher")
class CommMatcher(GPEMatcher):
    def __init__(self, nlp, name: str, overwrite: bool=False):
        super().__init__(nlp, name, overwrite)
        self.gpe_shape = "community"

    def initialize(self, get_examples=None, nlp=None, comm_area_path: str=None):
        gpes = gpd.read_parquet(comm_area_path)['community_name']
        super().initialize(get_examples, nlp, gpes)        


@Language.factory("neighborhood_name_matcher")
class HoodNameMatcher(GPEMatcher):
    def __init__(self, nlp, name: str, overwrite: bool=False):
        super().__init__(nlp, name, overwrite)
        self.gpe_shape = "neighborhood"

    def initialize(self, get_examples=None, nlp=None, neighborhood_path: str=None):
        gpes = pd.read_parquet(neighborhood_path)['name']
        super().initialize(get_examples, nlp, gpes)


@Language.factory("side_matcher")
class SideMatcher(GPEMatcher):
    def __init__(self, nlp, name: str, overwrite: bool=False):
        super().__init__(nlp, name, overwrite)
        self.gpe_shape = "side"

    def initialize(self, get_examples=None, nlp=None):
        gpes = pd.Series(sides)
        super().initialize(get_examples, nlp, gpes)
    

@Language.component("street_vs_neighborhood")
def street_vs_neighborhood(doc: Doc):
    # Fix street names that are actually neighborhoods
    new_ents = []
    for idx, ent in enumerate(doc._.ents):
        if ent.label_ == "FAC" and (ent.end != len(doc._.ents)):
            next_token = doc[ent.end]
            # Must match [FAC] neighborhood
            if next_token.text == "neighborhood":
                new_ent = Span(doc, ent.start, ent.end, label="GPE")
                new_ent._.gpe_shape = "neighborhood"
                new_ents.append(new_ent)
            else:
                new_ents.append(ent)
        else:
            new_ents.append(ent)
    doc._.ents = new_ents
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
        doc._.ents = filter_spans(list(doc._.ents) + matches)
        return doc

class Comparison(Flag):
    LT = auto()
    LTE = auto()
    EQ = auto()
    WITHIN = auto()
    CONTAINS = auto()
    GTE = auto()
    GT = auto()
    OVERLAP = LTE | EQ | WITHIN | CONTAINS | GTE
    DISJOINT = LT | GT

def compare(a: Span, b: Span):
        if a.end <= b.start:
            return Comparison.LT
        elif a.start < b.start and b.start <= a.end and a.end < b.end:
            return Comparison.LTE
        elif a.start == b.start and a.end == b.end:
            return Comparison.EQ
        elif b.start <= a.start and a.end <= b.end:
            return Comparison.WITHIN
        elif a.start <= b.start and b.end <= a.end:
            return Comparison.CONTAINS
        elif b.start < a.start and a.start < b.end and b.end < a.end:
            return Comparison.GTE
        elif b.end <= a.start:
            return Comparison.GT
        else:
            raise RuntimeError("Missing case")

def test_compare():
    nlp = spacy.blank("en")
    doc = nlp("A B C D E")
    assert compare(doc[0:2], doc[3:]) == Comparison.LT
    assert compare(doc[0:2], doc[2:]) == Comparison.LT
    assert compare(doc[0:3], doc[2:]) == Comparison.LTE
    assert compare(doc[0:4], doc[1:]) == Comparison.LTE
    assert compare(doc[0:4], doc[2:3]) == Comparison.CONTAINS
    assert compare(doc[0:4], doc[2:4]) == Comparison.CONTAINS
    assert compare(doc[2:3], doc[0:4]) == Comparison.WITHIN
    assert compare(doc[2:4], doc[0:4]) == Comparison.WITHIN
    assert compare(doc[3:], doc[3:]) == Comparison.EQ
    assert compare(doc[2:], doc[0:3]) == Comparison.GTE
    assert compare(doc[1:], doc[0:3]) == Comparison.GTE
    assert compare(doc[3:], doc[0:2]) == Comparison.GT
    assert compare(doc[2:], doc[0:2]) == Comparison.GT

    
@Language.component("ent_merger")
def ent_merger(doc: Doc) -> Doc:
    A = sorted(doc.ents, key=lambda x: x.start)
    B = sorted(doc._.ents, key=lambda x: x.start)
    merged = []
        
    i, j = 0, 0
    while i < len(A) and j < len(B):
        a = A[i]
        b = B[j]
        if compare(a, b) in [Comparison.LT, Comparison.CONTAINS]:
            merged.append(a)
            i += 1
        elif compare(a, b) == Comparison.GT:
            merged.append(b)
            j += 1
        else:
            if compare(a, b) != Comparison.EQ:
                logger.warning("Replacing %s (%s) [%d,%d) with %s (%s,%s) [%d,%d]",
                           a.text, a.label_, a.start, a.end,
                           b.text, b.label_, b._.gpe_shape, b.start, b.end)
            merged.append(b)
            i += 1  # Skip this item from A
            j += 1
    
    # Add any remaining items from A
    while i < len(A):
        merged.append(A[i])
        i += 1
    
    # Add any remaining items from B
    while j < len(B):
        merged.append(B[j])
        j += 1

    doc.ents = filter_spans(sorted(merged, key=lambda x: x.start))
    doc._.ents = None
    return doc


def apply_ents(doc: Doc, new_ents: list[Span], component_name: str, gpe_shape: str):
    doc._.ents = filter_spans(list(doc._.ents) + new_ents)
    gpe_match = [m for m in new_ents if m in doc._.ents]
    for ent in gpe_match:
        ent._.gpe_shape = gpe_shape
        logger.debug("%s matched %s as (%s,%s)", 
                     component_name, 
                     ent.text, 
                     ent.label_, 
                     ent._.gpe_shape)
    return doc


@Language.component("block_matcher")
def block_matcher(doc: Doc):
    # TODO: This sees to not be working. Though NER seems to be making block matches so i'm confused.
    new_ents = []
    # for idx, ent in enumerate(doc.ents):
    #     if ent.label_ == "FAC" and ent.start >= 3 and idx >= 1:
    #         prev_ent = list(doc.ents)[idx-1]
    #         prev_tokens = doc[ent.start - 3: ent.start]
    #         # Must match [CARDINAL] block of [FAC]
    #         if (prev_tokens[2].text == "of" and prev_tokens[1].text == "block"
    #             and prev_ent.label_ == "CARDINAL" and prev_tokens[0].text == prev_ent.text):
    #             new_ent = Span(doc, ent.start - 3, ent.end, label=ent.label)
    #             new_ents.append(new_ent)
    for idx, ent in enumerate(doc._.ents):
        if ent.label_ == "FAC" and ent.start >= 3:
            prev_tokens = doc[ent.start - 3: ent.start]
            # Must match [CARDINAL] block of [FAC]
            if (prev_tokens[2].text == "of" and prev_tokens[1].text == "block"
                and re.match(r"\d+00", prev_tokens[0].text)):
                new_ent = Span(doc, ent.start - 3, ent.end, label=ent.label)
                new_ents.append(new_ent)
    return apply_ents(doc, new_ents, "block_matcher", gpe_shape="block")


@Language.component("intersection_matcher")
def intersection_matcher(doc: Doc):
    new_ents = []
    for idx, ent in enumerate(doc._.ents):
        if ent.label_ == "FAC" and ent.start >= 2 and idx >= 1:
            prev_ent = list(doc._.ents)[idx-1]
            prev_tokens = doc[ent.start - 2: ent.start]
            # Must match [STREET] and [STREET]
            if ((prev_tokens[1].text == "and" or prev_tokens[1].text == "&")
                and prev_ent.label_ == "FAC" and prev_tokens[0].text == prev_ent.text):
                new_ent = Span(doc, ent.start - 2, ent.end, label=ent.label)
                new_ents.append(new_ent)
    return apply_ents(doc, new_ents, "intersection_matcher", gpe_shape="cross")


if __name__ == "__main__":
    test_compare()