import pandas as pd
import re

def plotable_dataframe(df):
    renames = {key:r'$%s}$' % key.replace('_','_{')  for key in df.keys() if '_' in key}

    return df.rename(columns=renames, index=renames)