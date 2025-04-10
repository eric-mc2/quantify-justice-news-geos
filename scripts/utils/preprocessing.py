import pandas as pd
from functools import singledispatch
from typing import TypeAlias

from textacy import preprocessing as tp
from spacy.tokens import Doc
import spacy
from spacy.util import filter_spans

from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

Text: TypeAlias = pd.DataFrame | list[Doc]

def split_train_dev_test(data: Text,
                         train_path: str = None, 
                         dev_path: str = None, 
                         test_path: str = None,
                         train_frac = .8,
                         test_frac = .5,
                         stratify: list[str] = None) -> tuple[Text,Text,Text]:
    train, dev_test = _split(data, train_frac, stratify)
    dev, test = _split(dev_test, test_frac, stratify)
    if train_path:
        train.to_parquet(train_path)
    if dev_path:
        dev.to_parquet(dev_path)
    if test_path:
        test.to_parquet(test_path)
    return train, dev, test

@singledispatch
def _split(data: Text, frac: float, stratify: list[str] = None) -> tuple[Text,Text]:
    pass

@_split.register(list)
def _(data: list[Doc], frac: float, stratify: list[str] = None) -> tuple[list[Doc],list[Doc]]:
    left_idx = pd.Series(list(range(len(data)))).sample(frac=frac, random_state=3925)
    left, right = [], []
    for i, doc in enumerate(data):
        if i in left_idx:
            left.append(doc)
        else:
            right.append(doc)
    return (left, right)

@_split.register(pd.DataFrame)
def _(data: pd.DataFrame, frac: float, stratify: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify:
        df_items = data[stratify].drop_duplicates().sample(frac=frac, random_state=3925)
        splits = df_items.merge(data, how='outer', indicator=True)
        left = splits[splits._merge == 'both'].drop(columns='_merge')
        right = splits[splits._merge == 'right_only'].drop(columns='_merge')
    else:
        left = data.sample(frac=frac, random_state=3925)
        right = data.loc[data.index.difference(left.index)]
    return (left, right)

def normalize(df: pd.DataFrame, text_col: str = "text"):
    preproc = tp.make_pipeline(
        tp.normalize.unicode,
        tp.replace.urls,
        tp.replace.user_handles,
        tp.replace.emails,
        tp.normalize.whitespace,
    )
    df = df.dropna(subset=text_col)
    df = df.assign(**{text_col: df[text_col].str.replace('\n', ' ').apply(preproc)})
    return df

def fix_span_prelabels(loc_data, article_data):
    # Adjust columns for merge
    loc_data = loc_data.rename(columns={'id':'loc_id', 'text': 'loc_text'})
    article_data = article_data.rename(columns={'id':'article_id', 'text':'article_text'})
    logger.debug("Merging %d coded articles onto %d full text", 
                 loc_data.article_id.nunique(),
                 article_data.article_id.nunique())
    loc_data = loc_data.merge(article_data, on='article_id', how='inner')
    logger.debug("Got %d merged labels", len(loc_data))

    # Ensure consistent text normalization
    loc_data = loc_data.pipe(normalize, 'loc_text').pipe(normalize, 'article_text')
    
    # Save negative rows to add back later. Only need to fix positive rows.
    mask = loc_data[['loc_text','start','end']].isna().any(axis=1)
    mask |= (loc_data['loc_text'] == "") | (loc_data['start'] == loc_data['end'])
    no_spans = loc_data.loc[mask]
    logger.debug("Input text had %d entity, %d non-entity rows", sum(~mask), sum(mask))
    
    spans = loc_data.loc[~mask].pipe(_fix_span_indexes).pipe(_align_spacy_tokens)
    loc_data = pd.concat([no_spans, spans], ignore_index=True)
    loc_data = loc_data.filter(['loc_id','article_id','start_token','end_token','text_token'])
    loc_data = loc_data.rename(columns={'loc_id':'id', 'start_token':'start','end_token':'end', 'text_token':'text'})
    return loc_data

def _fix_span_indexes(loc_data):
    need_cols = ['article_id','start','end','loc_text','article_text']
    assert all([c in loc_data.columns for c in need_cols])

    # Now fix dtypes for convenience
    loc_data['start'] = loc_data['start'].astype(int)
    loc_data['end'] = loc_data['end'].astype(int)
    
    assert loc_data.apply(lambda x: x.loc_text in x.article_text, axis=1).all()
    assert all(loc_data.groupby('article_id').article_text.nunique()==1)

    sliced = loc_data.apply(lambda row: row.article_text[row.start:row.end], axis=1)
    logger.debug("%.2f of spans boundaries initially aligned", (sliced == loc_data.loc_text).mean())

    aligned = {}
    for article_id, block in loc_data.groupby('article_id'):
        spans = []
        for row in block.itertuples():
            # Article may have duplicate locations. Adjust search past first occurrence.
            search_starts = [x['start_clean'] for x in spans if x['loc_text'] == row.loc_text]
            search_start = 1 + max(search_starts) if search_starts else 0
            match = row.article_text.index(row.loc_text, search_start)
            span = dict(start_clean = match, 
                        end_clean = match + len(row.loc_text), 
                        loc_text = row.loc_text, 
                        loc_id = row.loc_id)
            spans.append(span)
        aligned[article_id] = spans
    
    aligned = (pd.Series(aligned)
               .explode()
               .apply(pd.Series)
               .reset_index(names='article_id'))
    logger.debug("Alignment processed %d spans", len(aligned))

    loc_data = loc_data.merge(aligned, how='left')

    # Double-check that all rows are aligned.
    sliced = loc_data.apply(lambda row: row.article_text[row.start_clean:row.end_clean], axis=1)
    assert all(sliced == loc_data.loc_text)
    return loc_data

def _align_spacy_tokens(loc_data):
    # Snap alignment to spacy tokenization
    nlp = spacy.blank('en')
    spans = {}
    for row in loc_data.itertuples():
        doc = nlp(row.article_text)
        span_start = sum([t.idx <= row.start_clean for t in doc])
        span_end = sum([row.end_clean <= t.idx + len(t) for t in doc])
        spans[row.Index] = doc[span_start:span_end]
    spans = pd.Series(spans, name='spans')
    logger.debug("Processed %d spans", len(spans))
    
    # Convert to df
    aligned = (pd.concat([loc_data['article_id'], spans], axis=1)
               .groupby('article_id', as_index=False)
               .agg({'spans': filter_spans})
               .explode('spans')
               .assign(start_token = spans.apply(lambda x: x.start_char),
                       end_token = spans.apply(lambda x: x.end_char),
                       text_token = spans.apply(lambda x: x.text))
                .filter(['start_token','end_token','text_token']))
    logger.debug("Filtered spans to %d spans", len(aligned))
    
    loc_data = loc_data.join(aligned)
    return loc_data
