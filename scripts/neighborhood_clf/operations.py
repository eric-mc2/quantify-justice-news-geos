import json
import spacy
from spacy.tokens import DocBin, Doc
from scripts.utils.logging import setup_logger
from scripts.utils import preprocessing as pre
from scripts.utils.labelstudio import extract as extract_ls

logger = setup_logger(__name__)

def join_sentences(in_path, base_model, out_path):
    model = spacy.load(base_model)
    sentences = DocBin().from_disk(in_path).get_docs(model.vocab)
    sentences = sorted(sentences, key=lambda d: (d.user_data['id'], d.user_data['sentence_idx']))
    articles = DocBin(store_user_data=True)
    article = [sentences[0]]
    for sentence in sentences[1:]:
        if article[-1].user_data['id'] == sentence.user_data['id']:
            article.append(sentence)
        else:
            articles.add(Doc.from_docs(article))
            article = [sentence]
    articles.add(Doc.from_docs(article)) # process last sentence
    articles.to_disk(out_path)

def split(in_path, base_model, train_path, dev_path, test_path):    
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    train, dev, test = pre.split_train_dev_test(docs)
    traindb, devdb, testdb = DocBin(), DocBin(), DocBin()
    for d in train:
        traindb.add(d)
    for d in dev:
        devdb.add(d)
    for d in test:
        testdb.add(d)
    traindb.to_disk(train_path)
    devdb.to_disk(dev_path)
    testdb.to_disk(test_path)
    return train, dev, test

def pre_annotate(in_path, base_model, out_path):
    model = spacy.load(base_model)
    docs = list(DocBin().from_disk(in_path).get_docs(model.vocab))
    # XXX Drops user info
    texts = [{'text': d['text']} for d in docs]
    with open(out_path, "w") as f:
        json.dump(texts, f)

def annotate(in_path, out_path):
    data = extract_ls(in_path)
    groupby = data.columns.drop('label').to_list()
    data = data.groupby(groupby, as_index=False)['label'].agg(tuple)
    data = data.rename(columns={'label':'multilabel'})
    data.to_json(out_path, lines=True, orient="records", index=False)
    return data
