"""
Deep Learning Models for NBA Game Outcome Prediction
Trains Neural Network, SkipNet, and ResNet models with comprehensive evaluation.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve
)
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models.deep_learning import NeuralNetwork, SkipNetMLP, ResNetTabular
from tqdm import tqdm
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using device: {device}')


class GameDataset(Dataset):
    """PyTorch Dataset for NBA game data."""

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_data():
    """Load and prepare the dataset."""
    logger.info(f"Loading data from {config.DATA_PATH}...")

    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {config.DATA_PATH}")

    data = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded {len(data)} games")

    features = data.drop(columns=['GAME_ID', 'GAME_DATE', 'win_away', 'win_home'])
    target = data['win_home']

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Target distribution: {target.value_counts().to_dict()}")

    # Scale the features for better model performance
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, target,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=target
    )

    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    # Create datasets
    train_dataset = GameDataset(X_train_tensor, y_train_tensor)
    val_dataset = GameDataset(X_val_tensor, y_val_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, val_loader, X_val_tensor, y_val_tensor


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name="Model"):
    """
    Train and evaluate a model with comprehensive tracking.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (CPU or GPU)
        num_epochs: Number of epochs to train
        model_name: String identifier for the model

    Returns:
        Tuple of (trained model, training history)
    """
    logger.info(f"Training {model_name}...")

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            optimizer.zero_grad()

            outputs = model(batch_features)
            # If the model returns a tuple (e.g., SkipNet), take the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_features, val_targets in val_loader:
                val_features, val_targets = val_features.to(device), val_targets.to(device)
                outputs = model(val_features)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, val_targets)
                val_loss += loss.item() * val_features.size(0)

                predicted = (outputs >= 0.5).float()
                total += val_targets.size(0)
                correct += (predicted == val_targets).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total

        # Save history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()

        # Log progress
        if (epoch + 1) % config.LOG_INTERVAL == 0:
            logger.info(
                f"{model_name} Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

    # Load best model
    model.load_state_dict(best_model_state)
    logger.info(f"{model_name} - Best validation accuracy: {best_val_acc:.4f}")

    return model, history


def evaluate_model(model, X_val, y_val, device, model_name):
    """
    Comprehensive model evaluation.

    Args:
        model: Trained PyTorch model
        X_val: Validation features (tensor)
        y_val: Validation labels (tensor)
        device: Device
        model_name: Name of model

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating {model_name}...")

    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        outputs = model(X_val)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        y_pred = (outputs >= 0.5).float()
        y_pred_proba = outputs

    # Convert to numpy
    y_true = y_val.cpu().numpy().ravel()
    y_pred_np = y_pred.cpu().numpy().ravel()
    y_pred_proba_np = y_pred_proba.cpu().numpy().ravel()

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred_np),
        'precision': precision_score(y_true, y_pred_np),
        'recall': recall_score(y_true, y_pred_np),
        'f1_score': f1_score(y_true, y_pred_np),
        'roc_auc': roc_auc_score(y_true, y_pred_proba_np)
    }

    logger.info(f"{model_name} Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics, y_pred_np, y_pred_proba_np


def plot_training_curves(history, model_name):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curves', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{model_name} - Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    filename = f"training_curves_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved training curves to {filepath}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(config.FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved confusion matrix to {filepath}")


def save_model(model, model_name):
    """Save trained model to disk."""
    filename = f"{model_name.lower().replace(' ', '_')}_best.pt"
    filepath = os.path.join(config.MODELS_DIR, filename)
    torch.save(model.state_dict(), filepath)
    logger.info(f"Saved model to {filepath}")


def save_metrics(metrics, model_name):
    """Save metrics to JSON file."""
    filename = f"{model_name.lower().replace(' ', '_')}_metrics.json"
    filepath = os.path.join(config.METRICS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Saved metrics to {filepath}")


def plot_roc_curves_all(y_true, nn_proba, skipnet_proba, resnet_proba):
    """Plot ROC curves for all DL models."""
    plt.figure(figsize=(10, 8))

    # Neural Network ROC
    fpr_nn, tpr_nn, _ = roc_curve(y_true, nn_proba)
    auc_nn = roc_auc_score(y_true, nn_proba)
    plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {auc_nn:.3f})', linewidth=2)

    # SkipNet ROC
    fpr_skip, tpr_skip, _ = roc_curve(y_true, skipnet_proba)
    auc_skip = roc_auc_score(y_true, skipnet_proba)
    plt.plot(fpr_skip, tpr_skip, label=f'SkipNet (AUC = {auc_skip:.3f})', linewidth=2)

    # ResNet ROC
    fpr_res, tpr_res, _ = roc_curve(y_true, resnet_proba)
    auc_res = roc_auc_score(y_true, resnet_proba)
    plt.plot(fpr_res, tpr_res, label=f'ResNet (AUC = {auc_res:.3f})', linewidth=2)

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Deep Learning Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(config.FIGURES_DIR, "roc_curves_dl_models.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ROC curves to {filepath}")


def main():
    """Main function to train and evaluate deep learning models."""
    logger.info("=" * 60)
    logger.info("Starting Deep Learning Model Training Pipeline")
    logger.info("=" * 60)

    # Create output directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    # Load data
    train_loader, val_loader, X_val, y_val = load_data()
    input_dim = X_val.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    criterion = nn.BCELoss()

    # Lists to store results
    all_metrics = {}
    all_probas = {}

    # ========================================
    # Train Neural Network
    # ========================================
    logger.info("=" * 60)
    logger.info("1. Training Neural Network")
    logger.info("=" * 60)

    model1 = NeuralNetwork(input_dim).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=config.LEARNING_RATE)

    trained_model1, history1 = train_model(
        model1, train_loader, val_loader,
        criterion, optimizer1, device,
        config.NUM_EPOCHS, model_name="Neural Network"
    )

    metrics1, y_pred1, y_proba1 = evaluate_model(trained_model1, X_val, y_val, device, "Neural Network")
    plot_training_curves(history1, "Neural Network")
    plot_confusion_matrix(y_val.numpy().ravel(), y_pred1, "Neural Network")
    save_model(trained_model1, "Neural Network")
    save_metrics(metrics1, "Neural Network")

    all_metrics['neural_network'] = metrics1
    all_probas['neural_network'] = y_proba1

    # ========================================
    # Train SkipNet
    # ========================================
    logger.info("=" * 60)
    logger.info("2. Training SkipNetMLP")
    logger.info("=" * 60)

    model2 = SkipNetMLP(input_dim).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=config.LEARNING_RATE)

    trained_model2, history2 = train_model(
        model2, train_loader, val_loader,
        criterion, optimizer2, device,
        config.NUM_EPOCHS, model_name="SkipNet"
    )

    metrics2, y_pred2, y_proba2 = evaluate_model(trained_model2, X_val, y_val, device, "SkipNet")
    plot_training_curves(history2, "SkipNet")
    plot_confusion_matrix(y_val.numpy().ravel(), y_pred2, "SkipNet")
    save_model(trained_model2, "SkipNet")
    save_metrics(metrics2, "SkipNet")

    all_metrics['skipnet'] = metrics2
    all_probas['skipnet'] = y_proba2

    # ========================================
    # Train ResNet
    # ========================================
    logger.info("=" * 60)
    logger.info("3. Training ResNetTabular")
    logger.info("=" * 60)

    model3 = ResNetTabular(input_dim=input_dim, num_classes=1, dropout_rate=config.DROPOUT_RATE).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=config.LEARNING_RATE)

    trained_model3, history3 = train_model(
        model3, train_loader, val_loader,
        criterion, optimizer3, device,
        config.NUM_EPOCHS, model_name="ResNet"
    )

    metrics3, y_pred3, y_proba3 = evaluate_model(trained_model3, X_val, y_val, device, "ResNet")
    plot_training_curves(history3, "ResNet")
    plot_confusion_matrix(y_val.numpy().ravel(), y_pred3, "ResNet")
    save_model(trained_model3, "ResNet")
    save_metrics(metrics3, "ResNet")

    all_metrics['resnet'] = metrics3
    all_probas['resnet'] = y_proba3

    # ========================================
    # Combined Visualizations and Metrics
    # ========================================
    logger.info("=" * 60)
    logger.info("Generating combined visualizations...")
    logger.info("=" * 60)

    y_true = y_val.numpy().ravel()
    plot_roc_curves_all(y_true, y_proba1, y_proba2, y_proba3)

    # Save combined metrics
    filepath = os.path.join(config.METRICS_DIR, "dl_models_comparison.json")
    with open(filepath, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"Saved combined metrics to {filepath}")

    logger.info("=" * 60)
    logger.info("Deep Learning Model Training Completed Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
