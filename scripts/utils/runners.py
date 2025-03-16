import subprocess
import time
import dagster as dg
import pandas as pd
import os
import sys
import re
from scripts.utils.config import Config

def cmd(command: str | list[str], time_fmt=None, env=None):
    if isinstance(command, str):
        command = re.sub(r"\s+", " ", command).split(" ")
    
    start = time.time()
    process = subprocess.Popen(command, 
                               shell=False, 
                               env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True,
                               encoding='utf-8',
                               bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    end = time.time()

    if time_fmt:
        print(time_fmt.format(end - start))

def _obj_dtype(values: pd.Series):
    return values.apply(type).value_counts().idxmax().__name__

def dg_table_schema(df):
    columns = []
    for col, dt in df.dtypes.items():
        if dt.name == 'object':
            dt_name = _obj_dtype(df[col])
        else:
            dt_name = dt.name
        columns.append(dg.TableColumn(col, dt_name))
    return dg.TableSchema(columns=columns)

def dagster_dev():
    env = os.environ.copy()
    env["DAGSTER_HOME"] = Config().get_file_path("dagster.dagster_home")
    args = sys.argv[1:]
    cmd("dagster dev".split(" " ) + args, env=env)