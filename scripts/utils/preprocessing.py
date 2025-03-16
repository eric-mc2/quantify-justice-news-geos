import pandas as pd
from textacy import preprocessing as tp

def split_train_dev_test(data: pd.DataFrame, 
                         train_path: str = None, 
                         dev_path: str = None, 
                         test_path:str = None):
    train = data.sample(frac=.8, random_state=3925)
    dev_test = data.loc[data.index.difference(train.index)]
    dev = dev_test.sample(frac=.5, random_state=3925)
    test = dev_test.loc[dev_test.index.difference(dev.index)]
    if train_path:
        train.to_parquet(train_path)
    if dev_path:
        dev.to_parquet(dev_path)
    if test_path:
        test.to_parquet(test_path)
    return train, dev, test

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