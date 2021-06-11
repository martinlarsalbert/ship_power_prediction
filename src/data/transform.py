import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def extend(data:pd.DataFrame)-> pd.DataFrame:
    """Extend the dataset by adding aggregated features.

    Args:
        data (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: extended data:
            'u_wind','v_wind' : ship fixed wind (not apparant wind) 
    """

    data = data.copy()

    ## Ship fixed:
    r = R.from_euler('z', data['HDG'], degrees=True)
    r2 = r.inv()
    
    # Wind:
    data['W_wind'] = 0
    data[['u_wind','v_wind']] = r2.apply(data[['U_wind','V_wind','W_wind']])[:,0:2]
    data.drop(columns=['W_wind'], inplace=True)


    # Wave
    data['wave_direction'] = data['D_wave'] - data['HDG']

    return data




    
