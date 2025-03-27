import pandas as pd
import geopandas as gpd

from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from spacy.language import Language
from spacy.matcher.phrasematcher import PhraseMatcher
from spacy.matcher.matcher import Matcher

from scripts.geoms.operations import sides

@Language.factory("gpe_matcher")
def create_gpe_matcher(nlp, name, comm_area_path=None, neighborhood_path=None):
    gpes = pd.concat([gpd.read_parquet(comm_area_path)['community_name'].rename('name'),
                        pd.read_csv(neighborhood_path)['name'],
                        pd.Series(sides)], ignore_index=True)
    gpes = gpes.str.split(",", expand=False).explode()
    gpes = gpes.str.title().drop_duplicates().sort_values()
    
    matcher = PhraseMatcher(nlp.vocab)
    patterns = list(nlp.tokenizer.pipe(gpes))
    matcher.add("GPE", patterns)

    def match_gpes(doc: Doc):
        matches = matcher(doc, as_spans=True)
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc
    
    return match_gpes
    
@Language.factory("street_matcher")
def register_street_matcher(nlp, name, street_name_path=None):
    street_names = pd.read_csv(street_name_path)
    street_names = street_names.filter(like='combined').melt()['value']
    street_names = street_names.str.title().drop_duplicates().sort_values()

    loc_matcher = PhraseMatcher(nlp.vocab)
    patterns = list(nlp.tokenizer.pipe(street_names))
    loc_matcher.add("FAC", patterns)

    def match_streets(doc: Doc):
        matches = loc_matcher(doc, as_spans=True)
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc
        
    return match_streets

@Language.factory("age_matcher")
def register_age_matcher(nlp, name):
    matcher = Matcher(nlp.vocab) # Matcher might not be the right thing here since it operates on tokens
    matcher.add("CARDINAL", [[{"TEXT": {"REGEX": r"\d+[ -]year[ -]old"}}]])
    
    def match_age(doc: Doc):
        matches = matcher(doc, as_spans=True)
        doc.ents = filter_spans(list(doc.ents) + matches)
        return doc

    return match_age

@Language.component("block_matcher")
def expand_street_blocks(doc: Doc):
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
def expand_intersections(doc: Doc):
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