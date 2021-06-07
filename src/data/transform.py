import pandas as pd
import numpy as np

def transform(raw_data:pd.DataFrame)-> pd.DataFrame:
    """Transform the dataset, removing/adding/transforming features
    Args:
        raw_data (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: 
            T : mean draught [m]
            trim : trim [m] positive when trimmed to aft
    """

    data = raw_data.copy()

    #Transform draught
    data['T'] = (data['T_aft'] + data['T_fwd'])/2
    data['trim'] = data['T_aft'] - data['T_fwd']
    data.drop(columns=['T_aft', 'T_fwd'], inplace=True)

    return data