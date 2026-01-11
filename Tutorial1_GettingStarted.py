#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:19:08 2020

Homework answers for lesson 4 of "Friends of Tracking" #FoT

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""
import sys, os

# set up initial path to data
DATADIR = '/PATH/TO/WHERE/YOU/SAVED/THE/SAMPLE/DATA'
DATADIR = r'C:\Users\lover\github\sample-data\data'

sys.path.append(DATADIR)
from importlib import reload  # Python 3.4+

import Metrica_IO as mio
import Metrica_Viz as mviz

# set up initial path to data
# DATADIR = '/PATH/TO/WHERE/YOU/SAVED/THE/SAMPLE/DATA'
game_id = 2 # let's look at sample match 2

# read in the event data
events = mio.read_event_data(DATADIR,game_id)

# count the number of each event type in the data
print( events['Type'].value_counts() )

# Bit of housekeeping: unit conversion from metric data units to meters
events = mio.to_metric_coordinates(events)

# Get events by team
home_events = events[events['Team']=='Home']
away_events = events[events['Team']=='Away']

# Frequency of each event type by team
home_events['Type'].value_counts()
away_events['Type'].value_counts()

# Get all shots
shots = events[events['Type']=='SHOT']
home_shots = home_events[home_events.Type=='SHOT']
away_shots = away_events[away_events.Type=='SHOT']

# Look at frequency of each shot Subtype
home_shots['Subtype'].value_counts()
away_shots['Subtype'].value_counts()

# Look at the number of shots taken by each home player
print( home_shots['From'].value_counts() )

# Get the shots that led to a goal
home_goals = home_shots[home_shots['Subtype'].str.contains('-GOAL')].copy()
away_goals = away_shots[away_shots['Subtype'].str.contains('-GOAL')].copy()

# Add a column event 'Minute' to the data frame
home_goals['Minute'] = home_goals['Start Time [s]']/60.

# Plot the first goal
fig,ax = mviz.plot_pitch()
ax.plot( events.loc[198]['Start X'], events.loc[198]['Start Y'], 'ro' )
ax.annotate("", xy=events.loc[198][['End X','End Y']],
            xytext=events.loc[198][['Start X','Start Y']],
            alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))

# plot passing move in run up to goal
mviz.plot_events( events.loc[190:198], indicators = ['Marker','Arrow'], annotate=True )

# plot passing move in run up to goal
mviz.plot_events( events.loc[189:197], indicators = ['Marker','Arrow'], annotate=True )


# Plot the third  goal
fig,ax = mviz.plot_pitch()
ax.plot( events.loc[1118]['Start X'], events.loc[1118]['Start Y'], 'ro' )
ax.annotate("", xy=events.loc[1118][['End X','End Y']],
            xytext=events.loc[1118][['Start X','Start Y']],
            alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))


# Plot the third  goal
fig,ax = mviz.plot_pitch()
ax.plot( events.loc[1118:1681]['Start X'], events.loc[1118:1681]['Start Y'], 'ro' )
ax.annotate("", xy=events.loc[1118:1681][['End X','End Y']],
            xytext=events.loc[1118:1681][['Start X','Start Y']],
            alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))


# plot passing events in run up to goal
# plot passing move in run up to goal

mviz.plot_events( events.loc[1116:1117], indicators = ['Marker','Arrow'], annotate=True )




#### TRACKING DATA ####

# READING IN TRACKING DATA
tracking_home = mio.tracking_data(DATADIR,game_id,'Home')
tracking_away = mio.tracking_data(DATADIR,game_id,'Away')

# Look at the column namems
print( tracking_home.columns )

# Convert positions from metrica units to meters
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

# Plot some player trajectories (players 11,1,2,3,4)
# We are analyzing first 60 seconds, 1500 frames
fig,ax = mviz.plot_pitch()
# Probably GK
ax.plot( tracking_home['Home_11_x'].iloc[:1500], tracking_home['Home_11_y'].iloc[:1500], 'r.', markersize=1)
ax.plot( tracking_home['Home_1_x'].iloc[:1500], tracking_home['Home_1_y'].iloc[:1500], 'b.', markersize=1)
ax.plot( tracking_home['Home_2_x'].iloc[:1500], tracking_home['Home_2_y'].iloc[:1500], 'g.', markersize=1)
ax.plot( tracking_home['Home_3_x'].iloc[:1500], tracking_home['Home_3_y'].iloc[:1500], 'k.', markersize=1)
ax.plot( tracking_home['Home_4_x'].iloc[:1500], tracking_home['Home_4_y'].iloc[:1500], 'c.', markersize=1)

# plot player positions at ,atckick-off
KO_Frame = events.loc[0]['Start Frame']
fig,ax = mviz.plot_frame( tracking_home.loc[KO_Frame], tracking_away.loc[KO_Frame] )

# Goals loc numbers 198, 1118, 1723

# fig,ax = mviz.plot_frame( tracking_home.loc[KO_Frame], tracking_away.loc[GOAL_2] )

# PLOT POSITIONS AT GOAL
fig,ax = mviz.plot_events( events.loc[198:198], indicators = ['Marker','Arrow'], annotate=True )
goal_frame = events.loc[198]['Start Frame']
fig,ax = mviz.plot_frame( tracking_home.loc[goal_frame], tracking_away.loc[goal_frame], figax = (fig,ax) )

# PLOT POSITIONS AT GOAL
fig,ax = mviz.plot_events( events.loc[1118:1118],
                          indicators = ['Marker','Arrow'], annotate=True )
goal_frame = events.loc[1118]['Start Frame']
fig,ax = mviz.plot_frame(tracking_home.loc[goal_frame],
                         tracking_away.loc[goal_frame], figax = (fig,ax) )


#####


runup = events.loc[1110:1118].copy()   # or whatever range you used

coords = ["Start X","Start Y","End X","End Y"]
print(runup[["Type","Subtype"] + coords].head(30))

bad = runup[coords].isna().any(axis=1)
print("Bad rows:", runup.loc[bad, ["Type","Subtype"] + coords])
print("dtypes:", runup[coords].dtypes)

# We must drop NAN rows apparently

idx = 1118
runup = events.loc[1110:idx].copy()
# rows that can support an Arrow
arrow_ok = runup.dropna(subset=["Start X","Start Y","End X","End Y"]).copy()
# fig, ax = mviz.plot_pitch()
mviz.plot_events(arrow_ok, indicators=["Marker","Arrow"], annotate=True, ax=ax)


# Plot the first goal
fig,ax = mviz.plot_pitch()
ax.plot( events.loc[idx]['Start X'], events.loc[idx]['Start Y'], 'ro' )
ax.annotate("", xy=events.loc[idx][['End X','End Y']],
            xytext=events.loc[idx][['Start X','Start Y']],
            alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))

runup = events.loc[1110:idx].copy()
# rows that can support an Arrow
arrow_ok = runup.dropna(subset=["Start X","Start Y","End X","End Y"]).copy()

# plot passing move in run up to goal
mviz.plot_events( arrow_ok, indicators = ['Marker','Arrow'], annotate=True )


def plot_runup(idx):
    # Plot each goal
    idx_offset = idx - 10
    fig,ax = mviz.plot_pitch()
    ax.plot( events.loc[idx]['Start X'], events.loc[idx]['Start Y'], 'ro' )
    ax.annotate("", xy=events.loc[idx][['End X','End Y']],
                xytext=events.loc[idx][['Start X','Start Y']],
                alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))

    runup = events.loc[idx_offset:idx].copy()
    # rows that can support an Arrow
    arrow_ok = runup.dropna(subset=["Start X","Start Y","End X","End Y"]).copy()

    # plot passing move in run up to goal
    mviz.plot_events( arrow_ok, indicators = ['Marker','Arrow'], annotate=True )

def plot_runup_v2(idx):
    # Plot each goal
    idx_offset = idx - 10


    # fig,ax = mviz.plot_pitch()
# =============================================================================
#     ax.plot( events.loc[idx]['Start X'], events.loc[idx]['Start Y'], 'ro' )
#     ax.annotate("", xy=events.loc[idx][['End X','End Y']],
#                 xytext=events.loc[idx][['Start X','Start Y']],
#                 alpha=0.6, arrowprops=dict(arrowstyle="->",color='r'))
# =============================================================================
    runup = events.loc[idx_offset:idx].copy()
    runup = runup[runup["Type"].isin(["PASS", "SHOT"])].copy()

    print(f"We have the following from {runup['From']}")
    # plot passing move in run up to goal
    mviz.plot_events(runup, indicators = ['Marker','Arrow'], annotate=True )





for idx in home_goals.index:
    plot_runup_v2(idx)


# Print out just a single player's shots
player_9_mask = home_shots['From'] == 'Player9'
# home_shots.loc[player_9_mask]

player_shots = home_shots.loc[player_9_mask].dropna(subset=["Start X","Start Y","End X","End Y"])
fig, ax = mviz.plot_pitch()
mviz.plot_events(player_shots, indicators=["Marker", "Arrow"],
                 annotate=False)

######################################
# Highlight the goal, mute the shots that are non-goals

player_shots = home_shots.loc[player_9_mask].dropna(
    subset=["Start X","Start Y","End X","End Y"]
).copy()

is_goal = player_shots["Subtype"].str.contains("-GOAL", na=False)
shots_nongoal = player_shots[~is_goal]
shots_goal    = player_shots[is_goal]

fig, ax = mviz.plot_pitch()

# layer 1: non-goals (fainter)
mviz.plot_events(
    shots_nongoal,
    figax=(fig, ax),
    indicators=["Marker", "Arrow"],
    annotate=False,
    alpha=0.25
)

# layer 2: goals (stronger)
mviz.plot_events(
    shots_goal,
    figax=(fig, ax),
    indicators=["Marker", "Arrow"],
    annotate=False,
    alpha=0.9
)


# Compute travel distance

import numpy as np
import pandas as pd
import re

def distance_covered_by_team(tracking: pd.DataFrame, team_prefix: str) -> pd.DataFrame:
    """
    Compute total distance covered (meters) for each player in a Metrica tracking dataframe.

    Parameters
    ----------
    tracking : pd.DataFrame
        Metrica tracking data AFTER conversion to metric coordinates (meters).
    team_prefix : str
        "Home" or "Away" (matches column prefixes like Home_1_x, Away_7_y).

    Returns
    -------
    pd.DataFrame with columns: player, distance_m, distance_km
    """
    # Find player ids from columns like Home_11_x
    pat = re.compile(rf"^{re.escape(team_prefix)}_(\d+)_x$")
    player_ids = sorted(
        {int(m.group(1)) for c in tracking.columns if (m := pat.match(c))}
    )

    rows = []
    for pid in player_ids:
        x = pd.to_numeric(tracking[f"{team_prefix}_{pid}_x"], errors="coerce").to_numpy()
        y = pd.to_numeric(tracking[f"{team_prefix}_{pid}_y"], errors="coerce").to_numpy()

        # stepwise deltas
        dx = np.diff(x)
        dy = np.diff(y)

        # distance per step
        step = np.sqrt(dx * dx + dy * dy)
        step = np.minimum(step, 2.0)  # cap at 2 meters per frame (tweak based on fps)

        # valid when both frames have positions
        valid = (~np.isnan(x[:-1]) & ~np.isnan(y[:-1]) &
                 ~np.isnan(x[1:])  & ~np.isnan(y[1:]))

        dist_m = float(step[valid].sum())
        rows.append({"player": f"{team_prefix}_{pid}", "distance_m": dist_m, "distance_km": dist_m / 1000})

    out = pd.DataFrame(rows).sort_values("distance_m", ascending=False).reset_index(drop=True)
    return out


home_dist = distance_covered_by_team(tracking_home, "Home")
away_dist = distance_covered_by_team(tracking_away, "Away")

print(home_dist.head(5))
print(away_dist.head(5))


# END