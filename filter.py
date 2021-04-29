import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
from string import punctuation
import time


def get_nan_features(df):
    #Takes data frame as input and the output is a data frame with the columns
    #that have nan (and how many nan)
    features = list(df.columns)

    missing_values = list()
    for i in range(len(features)):
        missing_values.append((sum(df.iloc[:, i].isnull() == True)))

    missing_values_df = pd.DataFrame(columns=features)
    missing_values_df.loc[0] = missing_values

    nan_features = missing_values_df[missing_values_df != 0].T.dropna().T

    return nan_features


def get_cat_features(df):
    #Return a list with the categorical features and the number of categories by feature
    cat_features = []
    for col in df.columns:
        if df[col].dtype == 'object':
            cat_features.append((col, len(df[col].unique())))
    return cat_features


def clean_Turf(df):
    #
    Turf = {'Field Turf': 'Artificial', 'A-Turf Titan': 'Artificial', 'Grass': 'Natural',
            'UBU Sports Speed S5-M': 'Artificial',
            'Artificial': 'Artificial', 'DD GrassMaster': 'Artificial', 'Natural Grass': 'Natural',
            'UBU Speed Series-S5-M': 'Artificial', 'FieldTurf': 'Artificial', 'FieldTurf 360': 'Artificial',
            'Natural grass': 'Natural', 'grass': 'Natural',
            'Natural': 'Natural', 'Artifical': 'Artificial', 'FieldTurf360': 'Artificial', 'Naturall Grass': 'Natural',
            'Field turf': 'Artificial',
            'SISGrass': 'Artificial', 'Twenty-Four/Seven Turf': 'Artificial', 'natural grass': 'Natural'}

    df['Turf'] = df['Turf'].map(Turf)
    return df


def clean_game_clock(df):
    GameClock_min = []
    GameClock_sec = []
    for iter in range(df.shape[0]):
        time = str(df['GameClock'][iter]).split(':')
        GameClock_min.append(int(time[0]))
        GameClock_sec.append(int(time[1]))

    df['GameClock_min'] = GameClock_min
    df['GameClock_sec'] = GameClock_sec
    df['GameClockSec'] = 15 * 60 - (np.array(df['GameClock_min'] * 60) + np.array(df['GameClock_sec']))
    df = df.drop(columns=['GameClock_min', 'GameClock_sec', 'GameClock'])
    return df


def check_same_abb(df, col_name_1, col_name_2):
    #get which categories in col_name_1 are in col_name_2
    bol_vector = []
    for i in list(df[col_name_1].unique()):
        if i in df[col_name_2].unique():
            bol_vector.append(1)
        else:
            bol_vector.append(0)
    df = pd.DataFrame({col_name_1: df[col_name_1].unique(), 'Y/N': bol_vector})

    return df


def clean_wind_direction(txt):
    if pd.isna(txt):
        return np.nan
    txt = str(txt)
    txt = txt.lower()
    txt = txt.replace('-', '')
    txt = txt.replace('/', '')
    txt = txt.replace('from', '')
    txt = txt.replace(' ', '')
    txt = txt.replace('north', 'n')
    txt = txt.replace('east', 'e')
    txt = txt.replace('south', 's')
    txt = txt.replace('west', 'w')
    return txt


def transform_abbr(df):
    dict_abbr = {'CLV': 'CLE', 'HST': 'HOU', 'ARZ': 'ARI', 'BLT': 'BAL'}
    # Dict to apply map function
    for abb in df['HomeTeamAbbr'].unique():
        dict_abbr[abb] = abb
        # Dentro del corchete key y abb es el value
    df['PossessionTeam'] = df['PossessionTeam'].map(dict_abbr)
    df['FieldPosition'] = df['FieldPosition'].map(dict_abbr)

    return df


def transform_wind_direction(txt):
    if pd.isna(txt):
        return np.nan

    if txt == 'n':
        return 0
    if txt == 'nne' or txt == 'nen':
        return 1 / 8
    if txt == 'ne':
        return 2 / 8
    if txt == 'ene' or txt == 'nee':
        return 3 / 8
    if txt == 'e':
        return 4 / 8
    if txt == 'ese' or txt == 'see':
        return 5 / 8
    if txt == 'se':
        return 6 / 8
    if txt == 'ses' or txt == 'sse':
        return 7 / 8
    if txt == 's':
        return 8 / 8
    if txt == 'ssw' or txt == 'sws':
        return 9 / 8
    if txt == 'sw':
        return 10 / 8
    if txt == 'sww' or txt == 'wsw':
        return 11 / 8
    if txt == 'w':
        return 12 / 8
    if txt == 'wnw' or txt == 'nww':
        return 13 / 8
    if txt == 'nw':
        return 14 / 8
    if txt == 'nwn' or txt == 'nnw':
        return 15 / 8
    return np.nan


def clean_game_weather(txt):
    if pd.isna(txt):
        return np.nan
    txt = str(txt)
    txt = txt.lower()
    txt = txt.replace('clouidy', 'cloudy')
    txt = txt.replace('coudy', 'cloudy')
    txt = txt.replace('party', 'cloudy')
    return txt


def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans *= 0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans * 3
    if 'sunny' in txt or 'sun' in txt:
        return ans * 2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2 * ans
    if 'snow' in txt:
        return -3 * ans
    return 0


def clean_stadium_type(txt):
    if pd.isna(txt):
        return txt
    txt = str(txt)
    txt = txt.lower()
    txt = txt.replace('indoors', 'indoor')
    txt = txt.replace('outdoors', 'outdoor')
    txt = txt.replace('ourdoor', 'outdoor')
    txt = txt.replace('outside', 'outdoor')
    txt = txt.replace('outdor', 'outdoor')
    txt = txt.replace('outddors', 'outdoor')
    txt = txt.replace('oudoor', 'outdoor')
    txt = txt.replace('retr. roof - closed', 'retr. roof-closed')
    txt = txt.replace('retr. roof closed', 'retr. roof-closed')
    txt = txt.replace('domed', 'dome')
    txt = txt.replace('dome, closed', 'domed, closed')
    txt = txt.replace('closed dome', 'domed, closed')
    txt = txt.replace('retr. roof - open  ', 'retr. roof-open')
    return txt


def get_rusher(df):
    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    # df = df.drop(columns=['NflId', 'NflIdRusher'])

    return df


def yards_line_from_0(df):
    Yardleft = df['YardLine']
    IsPossesionHome = pd.Series(df['PossessionTeam'] == df['HomeTeamAbbr'])
    # df['IsHomeTeamInPossession'] = Possesion
    IsFieldPositionHome = pd.Series(df['FieldPosition'] == df['HomeTeamAbbr'])

    for i in range(len(IsFieldPositionHome)):
        if (IsFieldPositionHome[i] and IsPossesionHome[i]) or (not IsPossesionHome[i] and IsFieldPositionHome[i]):
            Yardleft[i] = Yardleft[i]
        if (not IsFieldPositionHome[i] and not IsPossesionHome[i]) or (
                not IsFieldPositionHome[i] and IsPossesionHome[i]):
            Yardleft[i] = 100 - Yardleft[i]

    df['YardLeft'] = list(Yardleft)

    return df


def transform_height_metres(txt):
    if pd.isna(txt):
        return txt
    txt = str(txt)
    txt = txt.split('-')
    txt = float(txt[0]) * 0.3048 + float(txt[1]) * 0.0254

    txt = round(txt, 2)

    return txt


def handle_time(df):
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
    df['PlayerAge'] = df.apply(
        lambda row: ((row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds()) / (60 * 60 * 24 * 365.25), axis=1)

    # df = df.drop(columns=['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'])
    return df


def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1


def clean_wind(df):
    df['WindSpeed'] = df['WindSpeed'].apply(
        lambda x: str(x).lower().replace('mph', '').strip() if not pd.isna(x) else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(str(x).split('-')[0]) + int(str(x).split('-')[0])) / 2
    if not pd.isna(x) and '-' in str(x) else x)

    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(str(x).split()[-1]) + int(str(x).split()[0])) / 2
    if not pd.isna(x) and type(x) != float and 'gusts up to' in x else x)

    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)

    return df


def get_wind_direction(df):
    df['WindDirection'] = df['WindDirection'].apply(clean_wind_direction)
    df['WindDirection'] = df['WindDirection'].apply(transform_wind_direction)

    return df


def clean_weather(df):
    df['GameWeather'] = df['GameWeather'].apply(clean_game_weather).apply(map_weather)
    return df

def preprocessing(df):
    df = get_wind_direction(df)
    df['PlayerHeight'] = df['PlayerHeight'].apply(transform_height_metres)
    df = get_rusher(df)
    df = yards_line_from_0(df)
    df = clean_Turf(df)
    df = clean_game_clock(df)
    df = transform_abbr(df)
    df = handle_time(df)
    df['StadiumType'] = df['StadiumType'].apply(clean_stadium_type)
    df = clean_weather(df)
    #df.drop(columns=['Season', 'YardLine'])
    return df

def get_list_cat_features(df):
    cat_features = []
    for item in df.columns:
        if df[item].dtype == 'object':
            cat_features.append(item)
    return cat_features
def get_list_float_features(df):
    float_features = []
    for item in df.columns:
        if (df[item].dtype == 'float' or df[item].dtype == 'int'):
            float_features.append(item)
    return float_features