import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def extract(in_data):
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
    data_annot = data['annotations'].apply(pd.Series)
    data_annot = data_annot['result'].explode().apply(pd.Series)
    data_annot = data_annot['value'].apply(pd.Series)
    data_annot = data_annot['choices'].explode()

    # Add catch-all negative label
    data_annot.fillna('IRRELEVANT',inplace=True)
    data_annot.rename('label', inplace=True)

    data = data_input.join(data_annot)
    return data