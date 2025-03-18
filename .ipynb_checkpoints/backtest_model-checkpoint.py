import etl
import utils

import pandas as pd
import numpy as np
from typing import Literal

from constants import LEAGUES

import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(filename='backtesting.log', encoding='utf-8', level=logging.INFO)

PREDICTED_SEASONS = [2024, 2023, 2022, 2021]

def backtest_league_season(season: int, league: Literal['M', 'W']):
    """Back-testing wrapper to calculate Brier score for the given season & league.

    Parameters
    ----------
    season : int
        The season to backtest
    league : Literal["M", "W"]
        Whether to test men's or women's predictions.
    """

    results = etl.read_postseason_results(league=league, season=season)
    results.head()
    formatted_results = etl.match_postseason_format_to_predictions(results_df=results)
    predictions = etl.read_predictions(league=league, season=season)

    formatted_results['ID'] = formatted_results['ID'].astype(str)
    predictions['ID'] = predictions['ID'].astype(str)
    pred_obs_df = formatted_results.merge(predictions, on='ID')

    pred = pred_obs_df['pred'].values
    obs = pred_obs_df['Result'].values
    brier = utils.brier_score(predictions=pred, observed=obs)
    logging.info(f'{season} {league} Brier: {brier:2f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for the team-strength model.')
    parser.add_argument('--season', type=int, required=False, default=None, help='The season to run the model for')
    parser.add_argument('--backtest_all', action='store_true', help='Backtest all years listed at the top.')
    args = parser.parse_args()

    if args.backtest_all:
        for season in PREDICTED_SEASONS:
            for league in LEAGUES:
                backtest_league_season(season=season, league=league)

    else:
        season = args.season
        for league in LEAGUES:
            backtest_league_season(season=season, league=league)