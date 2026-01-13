"""
Configuration file for NBA betting prediction project.
Contains all hyperparameters, paths, and settings.
"""

# Data Collection Settings
SEASONS = [
    "2024-25",
    "2023-24",
    "2022-23",
    "2021-22",
    "2020-21"
]
API_RATE_LIMIT_SLEEP = 0.5  # seconds to wait between API calls
ROLLING_WINDOW = 5  # number of games for rolling statistics

# File Paths
DATA_DIR = "data"
DATA_PATH = "data/DATA.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"

# Model Training Settings
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Architecture Settings
DROPOUT_RATE = 0.3
NN_LAYERS = [256, 128, 64, 32, 16]  # Layer sizes for neural network

# Tree Model Settings
RF_PARAM_GRID = {
    'n_estimators': [1000, 2000],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

ADA_PARAM_GRID = {
    'n_estimators': [1000, 2000],
    'learning_rate': [0.01, 0.1, 1.0]
}

# Cross-validation settings
CV_FOLDS = 5
CV_SCORING = 'roc_auc'

# Logging settings
LOG_INTERVAL = 10  # Print training stats every N epochs
