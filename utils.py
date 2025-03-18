from typing import Literal
import os
import pandas as pd
import numpy as np
import constants as c

def get_detailed_results_path(league: Literal['M', 'W']) -> str:
    """Gets the path to read in the detailed regular season results for a given league.

    Parameters
    ----------
    league : Literal["M", "W"]
        Whether to read NCAAM or NCAAW data.

    Returns
    -------
    str
        A filepath to the detailed results file.
    """

    path = os.path.join(c.DATA_PATH, league + c.REG_SEASON_DETAILED)
    return path

def get_postseason_results_path(league: Literal['M', 'W']) -> str:
    """Gets the path to read in the postseason results for a given league.

    Parameters
    ----------
    league : Literal["M", "W"]
        Whether to read NCAAM or NCAAW data.

    Returns
    -------
    str
        A filepath to the postseason results file.
    """
    path = os.path.join(c.DATA_PATH, league + c.POSTSEASON_RESULTS)
    return path

def save_team_ratings(ratings: pd.DataFrame, league: str, season: int):
    """Saves team ratings to a csv for a given league-season.

    Parameters
    ----------
    ratings : pd.DataFrame
        The model's ratings.
    league : str
        The league to model.
    season : int
        The season to model.
    """
    if not os.path.exists('ratings'):
        os.mkdir('ratings')
        
    filepath = os.path.join('ratings', league + "_" + str(season) + ".csv")
    ratings.to_csv(filepath, index=False)

def save_readable_predictions(labeled_preds: pd.DataFrame, league: str, season: int):
    """Saves a dataframe of "readable" predictions (i.e. those labelled with team names)

    Parameters
    ----------
    labeled_preds : pd.DataFrame
        The dataframe of readable predictions.
    league : str
        The league to predict for.
    season : int
        The season to predict for.
    """

    if not os.path.exists('predictions'):
        os.mkdir('predictions')
    filepath = os.path.join('predictions', league + "_" + str(season) + "Readable.csv")
    labeled_preds.to_csv(filepath, index=False)

def save_submission(submission: pd.DataFrame, league: str, season: int):
    """Saves a submission-formatted output.

    Parameters
    ----------
    submission : pd.DataFrame
        The dataframe of ID/Pred predictions.
    league : str
        The league to predict for.
    season : int
        The season to predict for.
    """
    
    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    filepath = os.path.join('predictions', league + "_" + str(season) + "Submission.csv")
    submission.to_csv(filepath, index=False)

def combine_and_save_full_submission(m_preds: pd.DataFrame, w_preds: pd.DataFrame, season: int):
    """Given men's and women's predictions, combines them and saves to a single file.

    Parameters
    ----------
    m_preds : pd.DataFrame
        The men's predictions.
    w_preds : pd.DataFrame
        The women's predictions.
    season : int
        The season to predict over.
    """

    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    filepath = os.path.join('predictions',  "Full" + str(season) + "Submission.csv")
    full_preds = pd.concat([m_preds, w_preds], ignore_index=True)
    full_preds.round({'Pred' : 3})
    full_preds.to_csv(filepath, index=False)



def format_match_id_from_teams(home: int, away: int, season: int) -> str:
    """Given two team IDs, returns the matchup ID for submission scoring purposes. 

    Parameters
    ----------
    home : int
        The home team's ID
    away : int
        The away team's ID
    season : int
        The season to predict over.

    Returns
    -------
    str
        The required output string formatted for competition submission.
    """
    if away > home:
        output_str = str(season) + "_" + str(int(home)) + "_" + str(int(away))
    else:
        output_str = str(season) + "_" + str(int(away)) + "_" + str(int(home))

    return output_str

def brier_score(predictions: np.array, observed: np.array) -> float:
    """Computes the Brier Score (MSE for booleans) over an array of predictions & observed.

    Parameters
    ----------
    predictions : np.array
        The predicted values. Should be bounded on [0,1].
    observed : np.array
        The observations, which should be booleans in the set (0, 1)

    Returns
    -------
    float
        The Brier score of the predictions vs. the observations. 
    """

    assert predictions.shape == observed.shape
    brier = np.mean((predictions - observed) ** 2)
    return brier