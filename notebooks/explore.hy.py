'''
Marc-Etienne Brunet
2021-12-21

Data Analysis for CSC 2558 class project.

This script is written to be run with the Hyrdogen plugin for the Atom text editor.
It can be converted to a Jupyter notebook by treating the '# %%' markers as cell boundaries.

For more information see: https://atom.io/packages/hydrogen

'''

# %% imports
import os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ranksums

# %% This allows us to make changes to code in ./src and use it on the go
# %load_ext autoreload
# %autoreload 2

# %% Import custom utils
import utils
utils.get_foo()

# %% Globals
DATA_DIR = '~/Code/maia-analysis/data/bot_logs'


# %% Load A-B test data dump
ab_raw = pd.read_csv(os.path.join(DATA_DIR, 'ab_testing.csv.gz'), parse_dates=['timestamp'],
                     infer_datetime_format=True, compression='gzip')
ab_raw.sort_values('timestamp', inplace=True, ignore_index=True)  # Sort by timestamp
print('A-B test started:', ab_raw.timestamp[0].tz_convert('US/Eastern'))
print(len(ab_raw), 'rows')
ab_raw.head()

# %% Trim unecessary data
# We only care about 5 columns
ab_info = ab_raw[['timestamp', 'bot_name', 'game_id', 'opponent_name']].copy()
ab_info['treatment'] = [x == 'stochastic' for x in ab_raw['category_name']]

# We only care about 3 bots
bot_mask = (ab_info.bot_name == 'maia1')
bot_mask |= (ab_info.bot_name == 'maia5')
bot_mask |= (ab_info.bot_name == 'maia9')
ab_info = ab_info[bot_mask]

ab_info.reset_index(inplace=True, drop=True)
ab_info.head()

# %% Address simple duplicates
# There's a bug where the same entries can occur with differences in username capitalization
# but lichess ensures usersnames are unique in a case insensitive way, so we can convert to lower
ab_info['opponent_name'] = ab_info['opponent_name'].str.lower()
init_len = len(ab_info)

# drop all duplicate entries
ab_info.drop_duplicates(subset=['bot_name', 'game_id', 'opponent_name', 'treatment'],
                        keep='first', inplace=True, ignore_index=True)
print('dropped', init_len - len(ab_info), 'duplicates')

# %% Address concerning duplicates
# Game ids should be unique, but the above bug may cause treatment to change in rare cases
# have a look
game_duplicate_mask = ab_info.duplicated(subset=['game_id'], keep=False)
ab_info[game_duplicate_mask]

# %% Use the first treatment, as it is believed this is what the user experienced
ab_info.drop_duplicates(subset=['game_id'], keep='first', inplace=True, ignore_index=True)

# %% Have a look at the data collected
print(len(ab_info), 'games recorded in A-B test')
print('fraction treated:', ab_info.treatment.mean())
print('bots:')
print(ab_info.bot_name.value_counts())
ab_info.describe(datetime_is_numeric=True)


# %% Visualize typical amount of game play
games_played = ab_info['opponent_name'].value_counts()
plt.hist(games_played, bins=50)
plt.xlabel(f'games played (max: {games_played.max()})', fontsize=14)
plt.ylabel('player count (log)', fontsize=14)
plt.yscale('log', nonpositive='clip')
plt.tight_layout()
plt.savefig('results/games_played.png', dpi=300)
plt.show()


# %% Load bot events to add game length data
events_raw = pd.read_csv(os.path.join(DATA_DIR, 'bot_events.csv.gz'), parse_dates=['timestamp'],
                         infer_datetime_format=True, compression='gzip')
events_raw.sort_values('timestamp', inplace=True, ignore_index=True)  # these need to be sorted here
print(len(events_raw), 'rows')
events_raw.head()

# %% Build mask to trim data
cols = ['timestamp', 'bot_name', 'game_id', 'event', 'data']  # only columns we need
bot_mask = (events_raw.bot_name == 'maia1')
bot_mask |= (events_raw.bot_name == 'maia5')
bot_mask |= (events_raw.bot_name == 'maia9')

# %% Find game start events
start_events = events_raw[cols][bot_mask & (events_raw.event == 'game_stream_start')].copy()
# we expect some duplicates, keep only the first
start_events.drop_duplicates(subset=['bot_name', 'game_id', 'event'], keep='first',
                             inplace=True, ignore_index=True)
# make sure there are no strange duplicates
assert start_events.duplicated(subset=['game_id', 'event'], keep=False).sum() == 0
start_events.set_index('game_id', inplace=True)

# %% Find game end events
end_events = events_raw[cols][bot_mask & (events_raw.event == 'game_end')].copy()
# we expect some duplicates, keep only the last
end_events.drop_duplicates(subset=['bot_name', 'game_id', 'event'], keep='last',
                           inplace=True, ignore_index=True)
# make sure there are no strange duplicates
assert end_events.duplicated(subset=['game_id', 'event'], keep=False).sum() == 0
end_events.set_index('game_id', inplace=True)

# %% Join
game_info = start_events.join(end_events, how='outer', lsuffix='_start', rsuffix='_end')

# %% Make sure the only bot name mismatches occur where end bot are NaN
assert game_info[(game_info['bot_name_start'] !=
       game_info['bot_name_end'])]['bot_name_end'].isna().all()
game_info.drop(columns=['bot_name_end'], inplace=True)
game_info.rename(columns={'bot_name_start': 'bot_name'}, inplace=True)

# %% Add game speed
game_info['speed'] = game_info['data_start'].map(lambda x: json.loads(x)['speed'])
game_info.value_counts('speed')

# %% Add reason for end
def extract_game_end(x):
    if isinstance(x, str):
        return json.loads(x)['game_over']

    elif pd.isna(x):
        return None

    else:
        print(x)
        raise TypeError('unexpected type')


game_info['game_over'] = game_info['data_end'].map(extract_game_end)
game_info.value_counts('game_over')

# %% Gameover == False indicates the game is not actually over
game_info.loc[game_info['game_over'] == False, 'timestamp_end'] = pd.NaT

# %% Add game length
game_length = (game_info['timestamp_end'] - game_info['timestamp_start'])
assert (game_length.dropna() >= pd.Timedelta(0)).all()
game_info['length'] = game_length
game_info.head()

# %% Total play time
print('total play time:', game_length.sum())

# %% Mask for proper games
proper_endings = set(['mate', 'resign', 'outoftime', 'draw', 'stalemate'])
proper_end_mask = game_info.game_over.map(lambda x: x in proper_endings)
game_info['proper_end'] = proper_end_mask
proper_end_mask.mean()

# %% Plot game lengths
mask = proper_end_mask & (game_length <= pd.Timedelta(60, 'm'))
plt.hist(game_length[mask].dropna().view(dtype=int) / 60e9, bins=100)
plt.xlabel('game length (min)', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.show()

# %% consider longer games
mask = proper_end_mask & (game_length > pd.Timedelta(60, 'm'))
plt.hist(game_length[mask].dropna().view(dtype=int) / 3600e9, bins=100)
plt.xlabel('game length (hrs)')
plt.ylabel('count')
plt.show()

# %% Join to A-B test data
ab_ext = ab_info.join(game_info[['length', 'game_over', 'speed', 'proper_end']], on='game_id')
ab_ext.head()

# %% Drop any games with incomplete information (principally those which are unfinished)
print('dropping', len(ab_ext) - len(ab_ext.dropna()), 'incomplete games')
ab_ext.dropna(inplace=True)
print(len(ab_ext), 'games,', ab_ext.proper_end.sum(), 'proper')

# %% Figure out treatment period boundary (timezone makes it a little unclear)
# Identify a few users with "sessions" that cross treatment boundaries
# ab_ext.value_counts('opponent_name')[ab_ext.value_counts('opponent_name') < 200]
# pd.options.display.max_rows = 200
# ab_ext.loc[ab_ext.opponent_name == 'peakay', ['timestamp', 'treatment']]
# ab_ext.loc[ab_ext.opponent_name == 'doc_holliday_thirty', ['timestamp', 'treatment']]
# ab_ext.loc[ab_ext.opponent_name == 'neltew', ['timestamp', 'treatment']]

# Seems like the timestamps are correctly localized to UTC, but a few treatment periods fall on
# day boundaries of US/Eastern, this is likely related to the case sensentive naming bug
ab_ext['timestamp'] = ab_ext['timestamp'].dt.tz_convert('US/Eastern')
ab_ext['date'] = ab_ext['timestamp'].dt.date


# %% Create the main DataFrame needed for analysis
by_user_date = ab_ext[ab_ext.proper_end].groupby(['opponent_name', 'date'])
df = pd.DataFrame({'games_played': by_user_date['game_id'].count(),
                   'time_played': by_user_date['length'].sum(),
                   'frac_treated': by_user_date['treatment'].mean()})
print(len(df), 'user-days')
df.head()


# %% Have a look at the fractional treatments
# These seem to be relted to the username case sensitivity bug, consider them treated
partial_treatment_mask = (df.frac_treated > 0) & (df.frac_treated < 1)
df[partial_treatment_mask]
##################################################################################################

# %% See how much data we're working with
print('num user-days', len(df))
print('treated:', (df.frac_treated > 0).sum())
print('untreated:', (df.frac_treated == 0).sum())

# %% Plot the treatment effect
games_mask = True  # (df.games_played > 0) & (df.games_played < 50)
mean_games_treated = df[games_mask & (df.frac_treated > 0)].games_played.mean()
print('mean treated:', mean_games_treated)
mean_games_untreated = df[games_mask & (df.frac_treated == 0)].games_played.mean()
print('mean untreated:', mean_games_untreated)
plt.hist(df[games_mask & (df.frac_treated > 0)].games_played,
         alpha=0.7, label=f'treated (mean={mean_games_treated:.3f})', bins=50, density=True)
plt.hist(df[games_mask & (df.frac_treated == 0)].games_played,
         alpha=0.7, label=f'untreated (mean={mean_games_untreated:.3f})', bins=50, density=True)
plt.xlabel('games played per user-day', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# %%
time_mask = True  # df.time_played < pd.Timedelta(8, 'h')
mean_time_treated = df[time_mask & (df.frac_treated > 0)].time_played.mean().total_seconds() / 60
print('mean treated:', mean_time_treated)
mean_time_untreated = df[time_mask & (df.frac_treated == 0)].time_played.mean().total_seconds() / 60
print('mean untreated:', mean_time_untreated)
plt.hist(df[time_mask & (df.frac_treated > 0)].time_played.view(dtype=int) / 1e9,
         alpha=0.7, label=f'treated (mean={mean_time_treated:.2f} min)', bins=50, density=True)
plt.hist(df[time_mask & (df.frac_treated == 0)].time_played.view(dtype=int) / 1e9,
         alpha=0.7, label=f'untreated (mean={mean_time_untreated:.2f} min)', bins=50, density=True)
plt.xlabel('time played per user-day (seconds)', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# %% Run a U-test to check the difference in these distributions
treated = df[df.frac_treated > 0].games_played.to_numpy()
untreated = df[df.frac_treated == 0].games_played.to_numpy()
print('Games played U-test:', ranksums(untreated, treated))

# %% Also check difference in time played
treated = df[df.frac_treated > 0].time_played.to_numpy().view(dtype=int)
untreated = df[df.frac_treated == 0].time_played.to_numpy().view(dtype=int)
print('Time played U-test:', ranksums(untreated, treated))

# %% Consider prior treatments
len(df)
df['prior_treatment'] = 0.0
for user in df.reset_index().opponent_name.unique():
    df.loc[user, 'prior_treatment'] = df.loc[user, 'frac_treated'].cumsum().to_numpy()

df.head(20)

# %% Repeat analysis for only first time treatments
prev_untreat_mask = df.prior_treatment <= df.frac_treated
print('previously untreated user-days:', len(df[prev_untreat_mask]))
print('treated:', len(df[prev_untreat_mask & (df.frac_treated > 0)]))
print('untreated:', len(df[prev_untreat_mask & (df.frac_treated == 0)]))

# %% Again run a U-test to check the difference in these distributions
treated = df[prev_untreat_mask & (df.frac_treated > 0)].games_played.to_numpy()
untreated = df[prev_untreat_mask & (df.frac_treated == 0)].games_played.to_numpy()
print('Games played U-test:', ranksums(untreated, treated))

# %% Again, also check difference in time played
treated = df[prev_untreat_mask & (df.frac_treated > 0)].time_played.to_numpy().view(dtype=int)
untreated = df[prev_untreat_mask & (df.frac_treated == 0)].time_played.to_numpy().view(dtype=int)
print('Time played U-test:', ranksums(untreated, treated))


# %%
####################################################################################################
# scratch code
####################################################################################################
# %% Add time until next game
time_until_next = pd.Series([pd.Timedelta.max for i in range(len(df))])

for user in df['opponent_name'].drop_duplicates(keep='first'):
    idx_mask = df.index[df['opponent_name'] == user]

    for i in range(len(idx_mask) - 1):
        diff = df['timestamp_start'][idx_mask[i+1]] - df['timestamp_end'][idx_mask[i]]
        time_until_next[idx_mask[i]] = diff

    time_until_next[idx_mask[-1]] = pd.NaT  # mark last game as NaT

df['time_until_next'] = time_until_next

# %%
(time_until_next.view(dtype=int) < 0).mean()

# %%
plt.hist(time_until_next.dropna().view(dtype=int) / 1e9,
         bins=50)
plt.xlabel('seconds until next game')
plt.ylabel('count')
plt.show()


# %% Load chat data
chat_df = pd.read_csv(os.path.join(DATA_DIR, 'chat_posts.csv.gz'), compression='gzip')
print(len(chat_df), 'rows')
chat_df.head()
