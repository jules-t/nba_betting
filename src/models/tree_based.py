"""
Tree-Based Models for NBA Game Outcome Prediction
Trains Random Forest and AdaBoost classifiers with hyperparameter tuning.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, classification_report
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load and prepare the dataset."""
    logger.info(f"Loading data from {config.DATA_PATH}...")

    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {config.DATA_PATH}")

    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded {len(df)} games")

    # Drop non-feature columns
    X = df.drop(['GAME_ID', 'GAME_DATE', 'win_away', 'win_home'], axis=1)
    y = df['win_home']

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance and return metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for logging

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Log metrics
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved confusion matrix to {filepath}")


def plot_feature_importance(model, feature_names, model_name):
    """Plot and save feature importance."""
    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"{model_name} does not have feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance - {model_name}')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()

    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved feature importance to {filepath}")


def save_model(model, model_name):
    """Save trained model to disk."""
    filename = f"{model_name.lower().replace(' ', '_')}_best.pkl"
    filepath = os.path.join(config.MODELS_DIR, filename)
    joblib.dump(model, filepath)
    logger.info(f"Saved model to {filepath}")


def save_metrics(metrics, model_name):
    """Save metrics to JSON file."""
    filename = f"{model_name.lower().replace(' ', '_')}_metrics.json"
    filepath = os.path.join(config.METRICS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Saved metrics to {filepath}")


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with GridSearchCV."""
    logger.info("=" * 60)
    logger.info("Training Random Forest Classifier")
    logger.info("=" * 60)

    rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
    rf_grid = GridSearchCV(
        rf,
        config.RF_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search
    rf_grid.fit(X_train, y_train)

    logger.info(f"Best parameters: {rf_grid.best_params_}")
    logger.info(f"Best CV ROC-AUC: {rf_grid.best_score_:.4f}")

    # Get best model
    best_rf = rf_grid.best_estimator_

    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(best_rf, X_test, y_test, "Random Forest")

    # Add hyperparameters to metrics
    metrics['best_params'] = rf_grid.best_params_
    metrics['best_cv_score'] = rf_grid.best_score_

    # Visualizations
    plot_confusion_matrix(y_test, y_pred, "Random Forest")
    plot_feature_importance(best_rf, X_train.columns, "Random Forest")

    # Save model and metrics
    save_model(best_rf, "Random Forest")
    save_metrics(metrics, "Random Forest")

    return best_rf, metrics, y_pred_proba


def train_adaboost(X_train, y_train, X_test, y_test):
    """Train AdaBoost with GridSearchCV."""
    logger.info("=" * 60)
    logger.info("Training AdaBoost Classifier")
    logger.info("=" * 60)

    ada = AdaBoostClassifier(random_state=config.RANDOM_STATE)
    ada_grid = GridSearchCV(
        ada,
        config.ADA_PARAM_GRID,
        cv=config.CV_FOLDS,
        scoring=config.CV_SCORING,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search
    ada_grid.fit(X_train, y_train)

    logger.info(f"Best parameters: {ada_grid.best_params_}")
    logger.info(f"Best CV ROC-AUC: {ada_grid.best_score_:.4f}")

    # Get best model
    best_ada = ada_grid.best_estimator_

    # Evaluate on test set
    metrics, y_pred, y_pred_proba = evaluate_model(best_ada, X_test, y_test, "AdaBoost")

    # Add hyperparameters to metrics
    metrics['best_params'] = ada_grid.best_params_
    metrics['best_cv_score'] = ada_grid.best_score_

    # Visualizations
    plot_confusion_matrix(y_test, y_pred, "AdaBoost")
    plot_feature_importance(best_ada, "AdaBoost")

    # Save model and metrics
    save_model(best_ada, "AdaBoost")
    save_metrics(metrics, "AdaBoost")

    return best_ada, metrics, y_pred_proba


def plot_roc_curves(y_test, rf_proba, ada_proba):
    """Plot ROC curves for both models."""
    plt.figure(figsize=(10, 8))

    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    auc_rf = roc_auc_score(y_test, rf_proba)
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', linewidth=2)

    # AdaBoost ROC
    fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_proba)
    auc_ada = roc_auc_score(y_test, ada_proba)
    plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {auc_ada:.3f})', linewidth=2)

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Tree-Based Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(config.FIGURES_DIR, "roc_curves_tree_models.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ROC curves to {filepath}")


def main():
    """Main function to train and evaluate tree-based models."""
    logger.info("Starting tree-based model training pipeline...")

    # Create output directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # Load data
    X, y = load_data()

    # Split data (single split for both models)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Train Random Forest
    rf_model, rf_metrics, rf_proba = train_random_forest(X_train, y_train, X_test, y_test)

    # Train AdaBoost
    ada_model, ada_metrics, ada_proba = train_adaboost(X_train, y_train, X_test, y_test)

    # Plot combined ROC curves
    plot_roc_curves(y_test, rf_proba, ada_proba)

    # Save combined metrics
    combined_metrics = {
        'random_forest': rf_metrics,
        'adaboost': ada_metrics
    }

    filepath = os.path.join(config.METRICS_DIR, "tree_models_comparison.json")
    with open(filepath, 'w') as f:
        json.dump(combined_metrics, f, indent=4)

    logger.info(f"Saved combined metrics to {filepath}")
    logger.info("=" * 60)
    logger.info("Tree-based model training completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
