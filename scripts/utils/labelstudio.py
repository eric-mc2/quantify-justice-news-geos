import pandas as pd
from dataclasses import dataclass
from scripts.utils.logging import setup_logger

logger = setup_logger(__name__)

def extract(in_data, task="textcat"):
    data = pd.read_json(in_data)
    cols = ['data','annotations','file_upload','created_at','updated_at','total_annotations','cancelled_annotations']
    data = data.filter(cols)

    # Only keep annotated rows (might not have manually labeled whole set yet.)
    logger.info("Filtering out {} un-annotated rows.".format(sum(data.total_annotations == 0)))
    data = data.loc[data.total_annotations > 0]

    # Annotations is a list so unpack it.
    data = data.explode('annotations')

    # Recover original data
    data_input = data['data'].apply(pd.Series)

    # Parse out the actual label. Discard other label provenance metadata.
    if task == "textcat":
        data_annot = data['annotations'].apply(pd.Series)
        data_annot = data_annot['result'].explode().apply(pd.Series)
        data_annot = data_annot['value'].apply(pd.Series)
        data_annot = data_annot['choices'].explode()

        # Add catch-all negative label
        data_annot.fillna('IRRELEVANT',inplace=True)
        data_annot.rename('label', inplace=True)
    elif task == "ner":
        data_annot = data['annotations'].apply(pd.Series)
        data_annot = data_annot['result'].explode().apply(pd.Series)
        data_ids = data_annot['id']
        data_annot = data_annot['value'].apply(pd.Series)
        data_annot = data_annot.drop(columns=0, errors='ignore')
        data_annot = pd.concat([data_ids, data_annot], axis=1)
        data_annot = data_annot.explode('labels')
        data_annot = data_annot.rename(columns={'text': 'ent_text', 
                                                'labels': 'label'})

    data = data_input.join(data_annot)
    return data

@dataclass
class LSValue:
    start: int
    end: int
    score: float
    text: str
    labels: list[str]

@dataclass
class LSResult:
    id: str
    from_name: str
    to_name: str
    type: str
    value: LSValue

@dataclass
class LSPrediction:
    model_version: str
    score: float
    result: list[LSResult]

@dataclass
class LSData:
    text: str

@dataclass
class LSDoc:
    data: LSData
    predictions: list[LSPrediction]