# NBA Game Outcome Prediction

A machine learning project that predicts NBA game outcomes using pre-game rolling statistics. This project explores both traditional tree-based models and deep learning architectures to forecast home team wins.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

**Can we predict NBA game outcomes using only pre-game features?**

This project uses rolling averages of team statistics (5-game windows) to predict whether the home team will win a game before it starts. The challenge is to extract meaningful patterns from historical performance data without access to real-time game information.

## Dataset

- **Source**: NBA API (`nba_api` library)
- **Time Period**: 2020-21 to 2024-25 seasons (5 complete seasons)
- **Total Games**: ~6,588 games across 30 NBA teams
- **Features**: 12 pre-game features (6 for home team, 6 for away team)
  - Rolling points (5-game average)
  - Rolling rebounds (5-game average)
  - Rolling assists (5-game average)
  - Rolling field goal percentage (5-game average)
  - Rest days since last game
  - Plus game metadata (GAME_ID, GAME_DATE)
- **Target Variable**: Binary classification - Home team win (1) or loss (0)
- **Class Balance**: ~50/50 split (home court advantage effect)

### Feature Engineering

The rolling statistics approach ensures that predictions use only information available before the game starts:
- For each game, we compute 5-game rolling averages from previous games
- Rest days capture fatigue and back-to-back game effects
- Separate features for home and away teams allow the model to learn matchup dynamics

## Models Implemented

### Tree-Based Models
1. **Random Forest Classifier**
   - Ensemble of decision trees with bagging
   - GridSearchCV for hyperparameter tuning
   - Provides feature importance analysis

2. **AdaBoost Classifier**
   - Boosted decision trees
   - Adaptive learning from misclassified examples
   - GridSearchCV optimization

### Deep Learning Models
3. **Neural Network**
   - 5-layer feedforward MLP
   - Architecture: Input → 256 → 128 → 64 → 32 → 16 → 1
   - BatchNormalization and Dropout (0.3) for regularization

4. **SkipNetMLP**
   - Custom architecture with gated skip connections
   - Learns to dynamically mix transformed and skip paths
   - 5 skip blocks with learnable gates

5. **ResNetTabular**
   - Residual network adapted for tabular data
   - 4 layers of residual blocks (64 → 128 → 256 → 512)
   - Addresses vanishing gradient problem in deep networks

## Results

*Results will be populated after model training completes*

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| AdaBoost | TBD | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD | TBD |
| SkipNet | TBD | TBD | TBD | TBD | TBD |
| ResNet | TBD | TBD | TBD | TBD | TBD |

### Key Insights

*Insights will be added after analysis*

## Project Structure

```
nba_betting/
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration and hyperparameters
│   ├── data_pipeline.py              # Data fetching and feature engineering
│   ├── training.py                   # Deep learning model training
│   └── models/
│       ├── __init__.py
│       ├── deep_learning.py          # PyTorch model architectures
│       └── tree_based.py             # Scikit-learn models
├── notebooks/                        # Jupyter notebooks
│   └── data_exploration.ipynb        # Exploratory data analysis
├── data/                             # Dataset
│   └── DATA.csv                      # Processed game data
├── models/                           # Saved trained models
│   ├── random_forest_best.pkl
│   ├── adaboost_best.pkl
│   ├── neural_network_best.pt
│   ├── skipnet_best.pt
│   └── resnet_best.pt
├── results/                          # Model outputs
│   ├── figures/                      # Visualizations
│   │   ├── confusion_matrix_*.png
│   │   ├── training_curves_*.png
│   │   ├── feature_importance_*.png
│   │   └── roc_curves_*.png
│   └── metrics/                      # Performance metrics
│       ├── *_metrics.json
│       └── *_comparison.json
├── tests/                            # Unit tests
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd nba_betting
```

2. **Create and activate a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection and Processing

Fetch NBA game data and engineer features:

```bash
python src/data_pipeline.py
```

This script:
- Fetches game data from NBA API for all 30 teams across 5 seasons
- Computes rolling statistics (5-game windows)
- Creates home/away perspectives
- Saves processed data to `data/DATA.csv`

**Note**: Data fetching takes ~10-15 minutes due to API rate limiting.

### 2. Train Tree-Based Models

Train Random Forest and AdaBoost with hyperparameter tuning:

```bash
python src/models/tree_based.py
```

This will:
- Perform GridSearchCV with 5-fold cross-validation
- Save best models to `models/` directory
- Generate confusion matrices and feature importance plots
- Save metrics to `results/metrics/`

**Training time**: ~20-40 minutes (depends on CPU cores)

### 3. Train Deep Learning Models

Train Neural Network, SkipNet, and ResNet:

```bash
python src/training.py
```

This will:
- Train all 3 DL models for 150 epochs each
- Track training/validation loss and accuracy
- Save best models based on validation accuracy
- Generate training curves, confusion matrices, and ROC curves
- Save metrics to `results/metrics/`

**Training time**: ~10-20 minutes (depends on GPU availability)

### 4. Explore the Data

Open the Jupyter notebook for exploratory data analysis:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 5. View Results

Check the `results/` directory for:
- **figures/**: All visualizations (PNG format)
- **metrics/**: JSON files with detailed performance metrics

## Configuration

All hyperparameters and settings are centralized in [src/config.py](src/config.py):

```python
# Key parameters
ROLLING_WINDOW = 5          # Rolling average window
BATCH_SIZE = 32             # Mini-batch size for DL
LEARNING_RATE = 0.001       # Adam optimizer learning rate
NUM_EPOCHS = 150            # Training epochs
TEST_SIZE = 0.2             # Train/test split ratio
DROPOUT_RATE = 0.3          # Dropout probability
```

Modify these values to experiment with different configurations.

## Technologies & Libraries

- **Python 3.9+**: Core programming language
- **PyTorch 2.0+**: Deep learning framework
- **scikit-learn**: Traditional ML algorithms and metrics
- **pandas & NumPy**: Data manipulation and numerical computing
- **nba_api**: Official NBA statistics API wrapper
- **matplotlib & seaborn**: Data visualization
- **tqdm**: Progress bars for long-running operations

## Key Features

- **Production-grade code**: Comprehensive error handling, logging, and type hints
- **Reproducible results**: Fixed random seeds and version-controlled dependencies
- **Comprehensive evaluation**: Multiple metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Rich visualizations**: Training curves, confusion matrices, ROC curves, feature importance
- **Model persistence**: Save and load trained models
- **Modular architecture**: Clean separation of concerns (data, models, training, evaluation)
- **Configuration management**: Centralized hyperparameters for easy experimentation

## Methodology

### Data Pipeline
1. Fetch historical game data from NBA API
2. Sort games chronologically by team
3. Compute rolling statistics using only past games
4. Calculate rest days between games
5. Create home/away perspectives for each game
6. Merge perspectives to create final dataset

### Model Training
1. Split data into train/test sets (80/20) with stratification
2. Feature scaling using StandardScaler (for DL models)
3. Hyperparameter tuning via GridSearchCV (tree models)
4. Train with validation monitoring (DL models)
5. Save best models based on validation performance
6. Comprehensive evaluation on test set

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: How many predicted wins were actual wins
- **Recall**: How many actual wins were predicted correctly
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (class discrimination ability)

## Future Improvements

- [ ] Add more features (player stats, injuries, historical matchups)
- [ ] Implement time series cross-validation (respecting temporal order)
- [ ] Try gradient boosting models (XGBoost, LightGBM, CatBoost)
- [ ] Ensemble methods (stacking, blending)
- [ ] Deploy as a web API with Flask/FastAPI
- [ ] Real-time predictions for upcoming games
- [ ] Betting strategy simulation with bankroll management
- [ ] Add confidence intervals for predictions
- [ ] Hyperparameter optimization with Optuna or Ray Tune

## Limitations

- **Feature scope**: Only uses basic team statistics, doesn't account for:
  - Individual player performance/injuries
  - Roster changes and trades
  - Coaching strategies
  - Weather (outdoor factors less relevant for NBA)
  - Referee assignments

- **Time horizon**: 5-game rolling window may not capture long-term trends

- **Home court advantage**: Model learns from historical data but doesn't explicitly model venue-specific effects

- **Temporal dynamics**: Games are treated independently, doesn't model momentum or streaks beyond rolling features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Jules Talbourdet**

- GitHub: [@julestalbourdet](https://github.com/julestalbourdet)
- LinkedIn: [Jules Talbourdet](https://linkedin.com/in/julestalbourdet)

## Acknowledgments

- NBA API developers for providing access to historical data
- scikit-learn and PyTorch communities for excellent documentation
- The machine learning community for research on sports analytics

---

**Note**: This project is for educational and research purposes. Please gamble responsibly if using predictions for betting purposes.
