from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NORMAL_LABELS = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb"}


@dataclass
class DataBundle:
    x_normal_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def load_rt_iot2022(dataset_path: str | Path, test_size: float = 0.2, seed: int = 42) -> DataBundle:
    """Load and preprocess RT_IOT2022 for state-based MaxEnt IRL."""
    df = pd.read_csv(dataset_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    cat_cols = [col for col in ("proto", "service") if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)

    df["is_attack"] = df["Attack_type"].apply(lambda x: int(x not in NORMAL_LABELS))

    normal_df = df[df["is_attack"] == 0].copy()
    attack_df = df[df["is_attack"] == 1].copy()

    normal_train, normal_test = train_test_split(normal_df, test_size=test_size, random_state=seed)
    attack_train, attack_test = train_test_split(attack_df, test_size=test_size, random_state=seed)

    drop_cols = [c for c in ("Attack_type", "is_attack") if c in df.columns]
    for part in (normal_train, normal_test, attack_train, attack_test):
        part.drop(columns=drop_cols, inplace=True)

    feature_names = normal_train.columns.tolist()

    numeric_cols = normal_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("id.orig_p", "id.resp_p")]

    scaler = StandardScaler()
    normal_train[numeric_cols] = scaler.fit_transform(normal_train[numeric_cols])

    for part in (normal_test, attack_train, attack_test):
        part[numeric_cols] = scaler.transform(part[numeric_cols])
        part.fillna(0, inplace=True)
        part.replace([np.inf, -np.inf], 0, inplace=True)

    x_normal_train = normal_train.values.astype(np.float32)
    x_test = np.vstack([normal_test.values, attack_test.values]).astype(np.float32)
    y_test = np.array([0] * len(normal_test) + [1] * len(attack_test), dtype=np.int32)

    return DataBundle(
        x_normal_train=x_normal_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )

