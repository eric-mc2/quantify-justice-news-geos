from spacy.language import Language
from spacy.tokens import Doc

@Language.component("entity_remapper")
def entity_remapper(doc: Doc):
    # Create mapping from original to new entity types
    loc_labels = [
        'FAC', # Buildings, airports, highways, bridges, etc.
        # 'ORG', # Companies, agencies, institutions, etc.
        'GPE', # Countries, cities, states
        'LOC', # Non-GPE locations, mountain ranges, bodies of water
        'EVENT' # Named events (e.g., "World Cup")
    ]

    mapping = {l: "NEWS_LOC" for l in loc_labels}
    
    for ent in doc.ents:
        ent.label_ = mapping.get(ent.label_, ent.label_)
    # # Apply mapping to entities
    # new_ents = []
    # for ent in doc.ents:
    #     if ent.label_ in mapping:
    #         # Create new entity with mapped label
    #         span = doc[ent.start:ent.end]
    #         span.label_ = mapping[ent.label_]
    #         new_ents.append(span)
    # Overwrite doc.ents with new entities
    # doc.ents = new_ents
    
    return doc

# Language.factory("entity_remapper", func=entity_remapper)