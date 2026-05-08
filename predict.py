from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data import load_rt_iot2022
from src.pipeline import MaxEntIRLConfig, TrainedModel, evaluate, predict_labels
from src.model import RewardNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained MaxEnt IRL model.")
    parser.add_argument("--data-path", type=str, default="../Data/RT_IOT/RT_IOT2022.csv")
    parser.add_argument("--model-dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)

    with open(model_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    config = MaxEntIRLConfig(**metadata["config"])
    network = RewardNetwork(hidden_units=tuple(config.hidden_units), dropout=config.dropout)
    network.build((None, metadata["input_dim"]))
    network.load_weights(str(model_dir / "reward_network.weights.h5"))

    trained = TrainedModel(network=network, threshold=float(metadata["threshold"]), config=config)
    data = load_rt_iot2022(args.data_path, seed=config.seed)
    y_pred, rewards = predict_labels(trained, data.x_test)

    metrics = evaluate(data.y_test, y_pred, rewards)
    print("Inference finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()

