from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .model import RewardNetwork, maxent_irl_loss


@dataclass
class MaxEntIRLConfig:
    hidden_units: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 256
    threshold_percentile: float = 5.0
    seed: int = 42


@dataclass
class TrainedModel:
    network: RewardNetwork
    threshold: float
    config: MaxEntIRLConfig


def train_maxent_irl(x_normal_train: np.ndarray, config: MaxEntIRLConfig) -> TrainedModel:
    tf.random.set_seed(config.seed)
    np.random.seed(config.seed)

    network = RewardNetwork(hidden_units=config.hidden_units, dropout=config.dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    dataset = tf.data.Dataset.from_tensor_slices(x_normal_train).shuffle(
        buffer_size=len(x_normal_train), seed=config.seed
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=False)

    for _ in range(config.epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                rewards = network(batch, training=True)
                loss = maxent_irl_loss(rewards)
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))

    normal_rewards = predict_rewards(network, x_normal_train)
    threshold = float(np.percentile(normal_rewards, config.threshold_percentile))
    return TrainedModel(network=network, threshold=threshold, config=config)


def predict_rewards(network: RewardNetwork, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    outputs = []
    for start in range(0, len(x), batch_size):
        batch = tf.convert_to_tensor(x[start : start + batch_size], dtype=tf.float32)
        rewards = network(batch, training=False).numpy().reshape(-1)
        outputs.append(rewards)
    return np.concatenate(outputs, axis=0)


def predict_labels(trained: TrainedModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rewards = predict_rewards(trained.network, x)
    y_pred = (rewards < trained.threshold).astype(np.int32)
    return y_pred, rewards


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, rewards: np.ndarray) -> Dict[str, float]:
    anomaly_scores = -rewards
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, anomaly_scores)),
    }

