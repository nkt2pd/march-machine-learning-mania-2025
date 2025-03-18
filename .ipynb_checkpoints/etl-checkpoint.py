import utils

from typing import Literal, Tuple, List, Dict
import pandas as pd
import numpy as np
import arviz as az
import constants as c
import os

def read_league_teams(league: Literal['M', 'W']) -> pd.DataFrame:
    """Imports the dataframe of teams.

    Parameters
    ----------
    league : Literal["M", "W"]
        Whether to read the NCAAM or NCAAW data.

    Returns
    -------
    pd.DataFrame
        A dataframe of team IDs
    """
    filepath = os.path.join(c.DATA_PATH, league + "Teams.csv")
    df = pd.read_csv(filepath)
    return df

def read_league_season_results(league: Literal['M', 'W'], season: int) -> pd.DataFrame:
    """Reads in detailed regular-season game results from a given season & league.

    Parameters
    ----------
    league : Literal["M", "W"]
        Whether to read the NCAAM or NCAAW data.
    season : int
        The season to read in data for.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the requested league-season data.
    """

    filepath = utils.get_detailed_results_path(league=league)
    df = pd.read_csv(filepath)
    df_season = df.query("Season == @season")
    return df_season

def read_postseason_results(league: Literal['M', 'W'], season: int) -> pd.DataFrame:
    """Reads in detailed regular-season game results from a given season & league.

    Parameters
    ----------
    league : Literal["M", "W"]
        Whether to read the NCAAM or NCAAW data.
    season : int
        The season to read in data for.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the requested league-season data.
    """
    filepath = utils.get_postseason_results_path(league=league)
    df = pd.read_csv(filepath)
    df_season = df.query('Season == @season')
    return df_season

def match_postseason_format_to_predictions(results_df: pd.DataFrame) -> pd.DataFrame:
    """Reformats a league-season dataframe in ID, Pred format.

    Parameters
    ----------
    results_df : pd.DataFrame
        The dataframe of results from `read_postseason_results`

    Returns
    -------
    pd.DataFrame
        A dataframe containing:
        - `ID`: keyed to match prediction ID.
        - `Result`: 1 if the 1st team in the ID pairing won and 0 otherwise. 
    """

    season = results_df['Season'].values[0]

    winner_ids = results_df['WTeamID'].values
    loser_ids = results_df['LTeamID'].values

    game_ids = []
    results = []

    for winner, loser in zip(winner_ids, loser_ids):
        game_id = f"{season}_{min(winner, loser)}_{max(winner, loser)}"
        # If the winning team ID is > the loser's, then it comes 2nd in the game ID.
        # Thus the result is a win for team 2 and a 0; vice versa for WTeamID < LTeamId.
        result = 0 if winner > loser else 1

        game_ids.append(game_id)
        results.append(result)

    output_df = pd.DataFrame(data={'ID' : game_ids, 'Result': results})
    return output_df

def read_predictions(season: int, league: Literal['M', 'W']) -> pd.DataFrame:
    """Reads in model predictions for a given season and league.

    Parameters
    ----------
    league : Literal["M", "W"]
        Whether to read the NCAAM or NCAAW data.
    season : int
        The season to read in data for.

    Returns
    -------
    pd.DataFrame
        An ID, Pred dataframe of predictions.
    """

    filepath = os.path.join(f'{season}_submission.csv')
    df = pd.read_csv(filepath)
    return df


def add_possessions(match_df: pd.DataFrame) -> pd.DataFrame:
    """Adds possession totals to each game. 

    Parameters
    ----------
    match_df : pd.DataFrame
        The dataframe of match results.

    Returns
    -------
    pd.DataFrame
        The same dataframe, with a column added matching the name c.POSSESSIONS.
    """

    fga = match_df["WFGA"] + match_df["LFGA"]
    fta = match_df["WFTA"] + match_df["LFTA"]
    to = match_df["WTO"] + match_df["LTO"]
    oreb = match_df["WOR"] + match_df["LOR"]

    match_df[c.POSSESSIONS] = (fga + 0.44 * fta + to - oreb) / 2
    return match_df

def add_efficiency(match_df: pd.DataFrame) -> pd.DataFrame:
    """Adds points-per-possession totals to the matches dataframe.

    Parameters
    ----------
    match_df : pd.DataFrame
        A dataframe of match results.

    Returns
    -------
    pd.DataFrame
        The same dataframe, with c.W_EFF and c.L_EFF added.
        These contain points-per-possession totals.
    """

    match_df[c.W_EFF] = match_df[c.W_PTS] / match_df[c.POSSESSIONS]
    match_df[c.L_EFF] = match_df[c.L_PTS] / match_df[c.POSSESSIONS]
    return match_df

def augment_game_results(game_results: pd.DataFrame) -> pd.DataFrame:
    """Wrapper around pre-processing steps to add necessary data.

    Parameters
    ----------
    game_results : pd.DataFrame
        A raw game results dataframe.

    Returns
    -------
    pd.DataFrame
        An augmented dataframe of results with possession totals and efficiency.
    """

    poss_df = add_possessions(game_results)
    eff_df = add_efficiency(poss_df)

    return eff_df

def pivot_win_loss_to_home_away(match_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots a dataframe of winner-loser results into a dataframe of home-away results.

    Parameters
    ----------
    match_df : pd.DataFrame
        The input dataframe of win-lose results.

    Returns
    -------
    pd.DataFrame
        An output dataframe of home-away results.
    """

    home_id = list()
    away_id = list()
    home_ppp = list()
    away_ppp = list()
    location = list()
    poss = list()

    for _ , row in match_df.iterrows():
        home_id.append(row[c.W_TEAM] if row[c.W_LOC] in ['H', 'N'] else row[c.L_TEAM])
        away_id.append(row[c.L_TEAM] if row[c.W_LOC] in ['H', 'N'] else row[c.W_TEAM])
        home_ppp.append(row[c.W_EFF] if row[c.W_LOC] in ['H', 'N'] else row[c.L_EFF])
        away_ppp.append(row[c.L_EFF] if row[c.W_LOC] in ['H', 'N'] else row[c.W_EFF])
        location.append(0 if row[c.W_LOC] == 'N' else 1)
        poss.append(row[c.POSSESSIONS])

    home_away_df = pd.DataFrame({
        c.HOME : home_id,
        c.AWAY : away_id,
        c.HOME_PPP : home_ppp,
        c.AWAY_PPP : away_ppp,
        c.LOCATION : location,
        c.POSSESSIONS : poss
    })
        
    return home_away_df

def encode_team_ids(match_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
    """Encodes team IDs as unique integers starting from 0.

    Parameters
    ----------
    match_df : pd.DataFrame
        The input match dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, np.array]
        A tuple:
        - the dataframe of encoded team IDs.
        - an array of uniques that can be used to de-index team IDs using .take() downstream.
    """

    team_ids = pd.concat([match_df[c.HOME], match_df[c.AWAY]], axis=0, ignore_index=True)
    _ , uniques = pd.factorize(team_ids)

    match_df[c.HOME] = [np.where(uniques == home_id)[0][0] for home_id in match_df[c.HOME]]
    match_df[c.AWAY] = [np.where(uniques == away_id)[0][0] for away_id in match_df[c.AWAY]]

    return match_df, uniques

def preprocess_match_data(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
    """Wrapper function for all dataframe preprocessing steps.

    Parameters
    ----------
    raw_data : pd.DataFrame
        The raw input data from a supplied data csv.

    Returns
    -------
    Tuple[pd.DataFrame, np.array]
        A tuple containing
        - the fully preprocessed dataframe
        - team labels for un-processing team IDs
    """

    processed_data = augment_game_results(raw_data)
    home_away = pivot_win_loss_to_home_away(processed_data)
    encoded_data, team_labels = encode_team_ids(home_away)

    encoded_data[c.HOME] = encoded_data[c.HOME].astype(int)
    encoded_data[c.AWAY] = encoded_data[c.AWAY].astype(int)
    encoded_data[c.LOCATION] = encoded_data[c.LOCATION].astype(int)

    return encoded_data, team_labels

def get_data_dict_and_coords(processed_data: pd.DataFrame, team_ids: np.array) -> Tuple[Dict[str, List[int | float]], Dict[str, np.array]]:
    """Given the processed model data, returns the necessary data dictionary and coordinates for PyMc.

    Parameters
    ----------
    processed_data : pd.DataFrame
        Dataframe piped through the preprocessing steps.
    team_ids : np.array
        The team IDs to include as the "teams" coordinate.

    Returns
    -------
    Tuple[Dict[str, List[int | float]], Dict[str, np.array]]
        The model data dictionary and coordinates.
    """

    output_dict = processed_data.to_dict("list")
    output_coords = {'teams' : team_ids}

    return output_dict, output_coords

def extract_team_ratings_from_trace(trace: az.InferenceData, league: Literal["M", "W"]) -> pd.DataFrame:
    """Given the sampled model, extracts team ratings.

    Parameters
    ----------
    trace : az.InferenceData
        The model trace.
    league : Literal["M", "W"]
        Whether the model is for men's or women's NCAA basketball.

    Returns
    -------
    pd.DataFrame
        A dataframe of team ratings.
    """

    ortg = az.summary(trace, var_names=['theta_off'], kind='stats').reset_index()
    drtg = az.summary(trace, var_names=['theta_def'], kind='stats').reset_index()

    ortg['team_id'] = ortg['index'].str.extract(r'(\d+)').astype(int)
    drtg['team_id'] = drtg['index'].str.extract(r'(\d+)').astype(int)

    team_ratings = ortg.merge(drtg, on='team_id', suffixes=('_off', '_def'))
    team_ratings['net_rating'] = team_ratings['mean_off'] + team_ratings['mean_def']

    team_names = read_league_teams(league=league)
    team_ratings = team_ratings.merge(team_names, left_on="team_id", right_on="TeamID")
    team_ratings = team_ratings.rename(columns={'mean_off' : 'off_rating', 'mean_def' : 'def_rating', 'TeamName' : 'team_name'})

    
    output_df = team_ratings[['team_id', 'team_name', 'net_rating', 'off_rating', 'def_rating']].sort_values('net_rating', ascending=False)
    rounded_df = output_df.round({'net_rating' : 3, 'off_rating' : 3, 'def_rating' : 3})

    return rounded_df

def add_team_names_to_ids(pred_data: pd.DataFrame, league: Literal["M", "W"]) -> pd.DataFrame:
    """Adds team names to a predicted dataframe.

    Parameters
    ----------
    pred_data : pd.DataFrame
        The predicted data.
    league : Literal["M", "W"]
        Whether the league is men's or women's NCAA

    Returns
    -------
    pd.DataFrame
        A readable dataframe with team names added alongside IDs.
    """
    
    team_names = read_league_teams(league=league)

    names_df = pred_data.merge(team_names, left_on=c.HOME, right_on="TeamID", suffixes=(None, '_home')).merge(team_names, left_on=c.AWAY, right_on="TeamID", suffixes=(None, "_away"))
    output_df = names_df[[c.HOME, c.AWAY, "TeamName", "TeamName_away", c.PREDICTION]]
    renamed_df = output_df.rename(columns={c.HOME : 'home_id', c.AWAY : 'away_id', "TeamName" : 'home_team', 'TeamName_away' : 'away_team', c.PREDICTION : 'home_wp'})

    return renamed_df