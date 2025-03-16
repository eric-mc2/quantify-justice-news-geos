import spacy

def load_spacy(model, **kwargs):
    try:
        nlp = spacy.load(model, **kwargs)
    except:
        spacy.cli.download(model)
        nlp = spacy.load(model, **kwargs)
    return nlp