"""
NBA Data Pipeline
Fetches historical NBA game data and engineers pre-game features for prediction.
"""

import os
import time
import logging
from typing import Optional, List, Dict
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import LeagueGameFinder
from nba_api.stats.static import teams
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_team_id(team_full_name: str) -> Optional[int]:
    """
    Get NBA team ID from team full name.

    Args:
        team_full_name: Full name of the NBA team (e.g., "Los Angeles Lakers")

    Returns:
        Team ID if found, None otherwise
    """
    try:
        nba_teams = teams.get_teams()
        team = next((team for team in nba_teams if team['full_name'] == team_full_name), None)
        if team is None:
            logger.warning(f"Team not found: {team_full_name}")
            return None
        return team['id']
    except Exception as e:
        logger.error(f"Error getting team ID for {team_full_name}: {e}")
        return None


def get_team_games(team_id: int, seasons: List[str]) -> Optional[pd.DataFrame]:
    """
    Retrieve historical game data for a team across multiple seasons.

    Args:
        team_id: NBA team ID
        seasons: List of season strings (e.g., ["2023-24", "2022-23"])

    Returns:
        DataFrame with game data, or None if retrieval fails
    """
    if team_id is None:
        logger.error("Cannot fetch games: team_id is None")
        return None

    lgf_dfs = []
    for season in seasons:
        try:
            # Retrieve historical game data for the team and season
            lgf = LeagueGameFinder(team_id_nullable=team_id, season_nullable=season)
            df_games = lgf.get_data_frames()[0]

            if df_games.empty:
                logger.warning(f"No games found for team {team_id} in season {season}")
                continue

            lgf_dfs.append(df_games)
            time.sleep(config.API_RATE_LIMIT_SLEEP)  # Pause to avoid rate limits

        except Exception as e:
            logger.error(f"Error fetching games for team {team_id} in season {season}: {e}")
            continue

    if not lgf_dfs:
        logger.error(f"No game data retrieved for team {team_id}")
        return None

    df_games = pd.concat(lgf_dfs, ignore_index=True)
    return df_games


def compute_rolling_features(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    """
    Compute rolling pre-game features for each game.

    For each game, computes rolling averages of key stats from the previous
    `window` games, and calculates days of rest since the previous game.

    Args:
        df: DataFrame with game data (must be sorted chronologically)
        window: Number of previous games to include in rolling average

    Returns:
        DataFrame with rolling features for each game
    """
    if window is None:
        window = config.ROLLING_WINDOW

    if df.empty:
        logger.warning("Empty DataFrame provided to compute_rolling_features")
        return pd.DataFrame()

    rolling_features = []
    for idx, row in df.iterrows():
        # All games before the current game
        previous_games = df.iloc[:idx]
        feature = {}

        if not previous_games.empty:
            window_games = previous_games.tail(window)
            feature["rolling_pts"] = window_games["PTS"].mean()
            feature["rolling_reb"] = window_games["REB"].mean()
            feature["rolling_ast"] = window_games["AST"].mean()
            feature["rolling_fg_pct"] = window_games["FG_PCT"].mean()
            last_game_date = previous_games["GAME_DATE"].iloc[-1]
            feature["rest_days"] = (row["GAME_DATE"] - last_game_date).days
        else:
            # For the first game, set default values to 0
            feature["rolling_pts"] = 0
            feature["rolling_reb"] = 0
            feature["rolling_ast"] = 0
            feature["rolling_fg_pct"] = 0
            feature["rest_days"] = 0

        feature["GAME_ID"] = row["GAME_ID"]
        rolling_features.append(feature)

    return pd.DataFrame(rolling_features)


def main() -> None:
    """
    Main function to fetch NBA data and create processed dataset.
    """
    logger.info("Starting NBA data pipeline...")

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Get all NBA teams
    try:
        nba_team_list = teams.get_teams()
        team_names = [team["full_name"] for team in nba_team_list]
        logger.info(f"Found {len(team_names)} NBA teams")
    except Exception as e:
        logger.error(f"Error fetching team list: {e}")
        return

    # Get team IDs
    team_ids: Dict[str, Optional[int]] = {name: get_team_id(name) for name in team_names}

    # Filter out teams with None IDs
    valid_team_ids = {name: tid for name, tid in team_ids.items() if tid is not None}
    logger.info(f"Processing {len(valid_team_ids)} teams with valid IDs")

    # Retrieve data for all teams
    data_teams = {}
    logger.info("Downloading data for all teams...")

    for team_name, team_id in tqdm(valid_team_ids.items(), desc="Processing Teams"):
        df_games = get_team_games(team_id, config.SEASONS)

        if df_games is None or df_games.empty:
            logger.warning(f"Skipping team {team_name}: no data retrieved")
            continue

        # Convert GAME_DATE to datetime and sort chronologically
        df_games["GAME_DATE"] = pd.to_datetime(df_games["GAME_DATE"])
        df_games = df_games.sort_values("GAME_DATE").reset_index(drop=True)

        # Compute rolling features for the team
        df_rolling = compute_rolling_features(df_games, window=config.ROLLING_WINDOW)

        # Merge the rolling features with the original game data
        df_model = pd.merge(df_games, df_rolling, on="GAME_ID", how="left")

        # Pre-game context features
        # Home/away indicator based on MATCHUP
        df_model["home_game"] = df_model["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)
        # Create target variable (win=1, loss=0)
        df_model["win"] = df_model["WL"].apply(lambda x: 1 if x == "W" else 0)

        # Keep only pre-game features
        features_to_keep = [
            "GAME_ID", "GAME_DATE", "home_game", "rolling_pts", "rolling_reb",
            "rolling_ast", "rolling_fg_pct", "rest_days", "win"
        ]
        df_model_final = df_model[features_to_keep].copy()
        data_teams[team_name] = df_model_final

    if not data_teams:
        logger.error("No data collected for any team. Exiting.")
        return

    logger.info(f"Successfully collected data for {len(data_teams)} teams")

    # Combine all team data into one master DataFrame
    df_all = pd.concat(data_teams.values(), ignore_index=True)
    logger.info(f"Total games collected: {len(df_all)}")

    # Create home and away datasets
    df_home = df_all[df_all["home_game"] == 1].copy()
    df_away = df_all[df_all["home_game"] == 0].copy()

    logger.info(f"Home games: {len(df_home)}, Away games: {len(df_away)}")

    # Rename feature columns for clarity
    features = ["rolling_pts", "rolling_reb", "rolling_ast", "rolling_fg_pct", "rest_days", "win"]
    df_home = df_home.rename(columns={col: col + "_home" for col in features})
    df_away = df_away.rename(columns={col: col + "_away" for col in features})

    # Merge home and away datasets
    # We merge on GAME_ID and GAME_DATE so that each row represents one game
    df_games_merged = pd.merge(
        df_home, df_away,
        on=["GAME_ID", "GAME_DATE"],
        how="inner"
    )

    logger.info(f"Merged dataset size: {len(df_games_merged)} games")

    # Remove redundant columns
    for col in ["home_game_home", "home_game_away"]:
        if col in df_games_merged.columns:
            df_games_merged.drop(columns=[col], inplace=True)

    # Validate final dataset
    if df_games_merged.isnull().any().any():
        logger.warning("Dataset contains null values")
        null_counts = df_games_merged.isnull().sum()
        logger.warning(f"Null counts per column:\n{null_counts[null_counts > 0]}")

    # Save the final merged dataset
    try:
        df_games_merged.to_csv(config.DATA_PATH, index=False)
        logger.info(f"Final merged dataset saved to {config.DATA_PATH}")
        logger.info(f"Dataset shape: {df_games_merged.shape}")
        logger.info(f"Columns: {list(df_games_merged.columns)}")
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return

    logger.info("Data pipeline completed successfully!")


if __name__ == "__main__":
    main()
