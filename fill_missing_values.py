
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import re
import scipy
import fancyimpute as fi


most_matches = pd.read_csv('most_seasons_unnormalized.csv')
recent_matches = pd.read_csv('recent_seasons_unnormalized.csv')

most_matches = most_matches.fillna(np.nan)
most_matches = most_matches.replace(-1, np.nan)

recent_matches = recent_matches.fillna(np.nan)
recent_matches = recent_matches.replace(-1, np.nan)

most_matches.drop(list(most_matches.filter(regex='Unnamed: ')), axis=1, inplace=True)
recent_matches.drop(list(recent_matches.filter(regex='Unnamed: ')), axis=1, inplace=True)

all_matches = most_matches.append(recent_matches, ignore_index=True)


betting_odds_home = ['B365H', 'BWH', 'IWH', 'LBH', 'PSH', 'WHH', 'SJH', 'VCH', 'GBH', 'BSH']
betting_odds_draw = ['B365D', 'BWD', 'IWD', 'LBD', 'PSD', 'WHD', 'SJD', 'VCD', 'GBD', 'BSD']
betting_odds_away = ['B365A', 'BWA', 'IWA', 'LBA', 'PSA', 'WHA', 'SJA', 'VCA', 'GBA', 'BSA']
betting_odds_all = [betting_odds_home, betting_odds_draw, betting_odds_away]
betting_odds_combined = betting_odds_home + betting_odds_draw + betting_odds_away

rows = all_matches[pd.isnull(all_matches[betting_odds_combined]).any(axis=1)]
print(len(rows))

for index, row in rows.iterrows():
    if index % 1000 == 0:
        print(index)
    for betting_odds in betting_odds_all:
        mean = np.mean(row[betting_odds])
        all_matches.loc[index, row[betting_odds].index[row[betting_odds].isnull().tolist()]] = mean


home = all_matches.filter(regex='__home_')
away = all_matches.filter(regex='__away_')

home_filled = fi.KNN().fit_transform(home)
home_filled = pd.DataFrame(data=home_filled, columns=home.columns, index=home.index)

away_filled = fi.KNN().fit_transform(away)
away_filled = pd.DataFrame(data=away_filled, columns=away.columns, index=away.index)

all_matches_filled = all_matches.copy()
all_matches_filled[home.columns] = home_filled
all_matches_filled[away.columns] = away_filled

cols_to_normalize = ['__home_buildUpPlaySpeed', '__home_buildUpPlayDribbling',
                     '__home_buildUpPlayPassing',
                     '__home_chanceCreationPassing', '__home_chanceCreationCrossing',
                     '__home_chanceCreationShooting', '__home_defencePressure',
                     '__home_defenceAggression', '__home_defenceTeamWidth',
                     '__away_buildUpPlaySpeed',
                     '__away_buildUpPlayDribbling', '__away_buildUpPlayPassing',
                     '__away_chanceCreationPassing', '__away_chanceCreationCrossing',
                     '__away_chanceCreationShooting', '__away_defencePressure',
                     '__away_defenceAggression', '__away_defenceTeamWidth']

all_matches_filled_normed = all_matches_filled.copy()

# Normalize columns
for column in cols_to_normalize:
    col = np.array(all_matches_filled[column].values.astype(np.float))
    all_matches_filled_normed[column] = (col - col.min()) / (col.max() - col.min())



all_matches_filled = all_matches_filled.fillna(-1)
all_matches_filled_normed = all_matches_filled_normed.fillna(-1)


most_matches_filled = all_matches_filled[:len(most_matches)]
most_matches_filled_normed = all_matches_filled_normed[:len(most_matches)]

recent_matches_filled = all_matches_filled[-len(recent_matches):]
recent_matches_filled_normed = all_matches_filled_normed[-len(recent_matches):]



most_matches_filled.to_csv('most_seasons_filled_unnormalized.csv', index=False)
recent_matches_filled.to_csv('recent_seasons_filled_unnormalized.csv', index=False)

most_matches_filled_normed.to_csv('most_seasons_filled_normalized.csv', index=False)
recent_matches_filled_normed.to_csv('recent_seasons_filled_normalized.csv', index=False)
