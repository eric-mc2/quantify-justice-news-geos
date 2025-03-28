import dagster as dg
import pandas as pd
import os
import sys
from scripts.utils.config import Config
from scripts.utils.runners import cmd

def dg_table_schema(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    columns = []
    for col, dt in df.dtypes.items():
        if dt.name == 'object':
            dt_name = _obj_dtype(df[col])
        else:
            dt_name = dt.name
        columns.append(dg.TableColumn(col, dt_name))
    return dg.TableSchema(columns=columns)

def _obj_dtype(values: pd.Series):
    return values.apply(type).value_counts().idxmax().__name__

def dagster_dev():
    env = os.environ.copy()
    env["DAGSTER_HOME"] = Config().get_file_path("dagster.dagster_home")
    args = sys.argv[1:]
    cmd("dagster dev".split(" " ) + args, env=env)
