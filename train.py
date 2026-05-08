from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data import load_rt_iot2022
from src.pipeline import (
    MaxEntIRLConfig,
    evaluate,
    predict_labels,
    predict_rewards,
    train_maxent_irl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train proposed state-based MaxEnt IRL model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../Data/RT_IOT/RT_IOT2022.csv",
        help="Path to RT_IOT2022.csv",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_rt_iot2022(args.data_path)
    config = MaxEntIRLConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threshold_percentile=args.threshold_percentile,
    )
    trained = train_maxent_irl(data.x_normal_train, config=config)

    y_pred, rewards = predict_labels(trained, data.x_test)
    metrics = evaluate(data.y_test, y_pred, rewards)

    trained.network.save_weights(str(output_dir / "reward_network.weights.h5"))
    normal_train_rewards = predict_rewards(trained.network, data.x_normal_train)
    np.save(output_dir / "normal_train_rewards.npy", normal_train_rewards)

    metadata = {
        "threshold": trained.threshold,
        "config": config.__dict__,
        "metrics": metrics,
        "n_train_normal": int(len(data.x_normal_train)),
        "n_test": int(len(data.x_test)),
        "input_dim": int(data.x_normal_train.shape[1]),
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    np.random.seed(42)
    tf.random.set_seed(42)
    main()

