from __future__ import annotations

import tensorflow as tf


class RewardNetwork(tf.keras.Model):
    """Feedforward reward model R_theta(s) used in the paper."""

    def __init__(self, hidden_units: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        self.blocks = []
        for units in hidden_units:
            self.blocks.append(tf.keras.layers.Dense(units, activation="relu"))
            self.blocks.append(tf.keras.layers.Dropout(dropout))
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, x, training: bool = False):
        h = tf.cast(x, tf.float32)
        for layer in self.blocks:
            if isinstance(layer, tf.keras.layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)
        return self.output_layer(h)


def maxent_irl_loss(rewards: tf.Tensor) -> tf.Tensor:
    """State-based MaxEnt IRL loss from the manuscript."""
    rewards = tf.reshape(rewards, (-1,))
    return -tf.reduce_mean(rewards - tf.reduce_logsumexp(rewards))

