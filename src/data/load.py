import pandas as pd
import numpy as np


def raw()->pd.DataFrame:
    """Load the data and do some renamings...

    Returns:
        pd.DataFrame: [description]
    """


    raw_data = pd.read_csv('../../data/raw/Project2_shipdata.csv')

    # Clean the descriptions:
    renames = {key:key.split(':')[0] for key in raw_data.columns}
    raw_data.rename(columns=renames, inplace=True)

    # Sign change
    raw_data['V']*=-1
    raw_data['T_aft']*=-1
    raw_data['T_fwd']*=-1
    

    return raw_data