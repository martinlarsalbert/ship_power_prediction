import pandas as pd
import numpy as np
import os.path

def raw()->pd.DataFrame:
    """Load the data and do some renamings...

    Returns:
        pd.DataFrame: [description]
    """

    data_path = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0],'data')
    file_path = os.path.join(data_path,'raw','Project2_shipdata.csv')
    raw_data = pd.read_csv(file_path)

    # Clean the descriptions:
    renames = {key:key.split(':')[0] for key in raw_data.columns}
    raw_data.rename(columns=renames, inplace=True)

    # Sign change
    raw_data['V']*=-1
    raw_data['T_aft']*=-1
    raw_data['T_fwd']*=-1
    

    return raw_data