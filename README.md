# MaxEnt IRL for IoT Cyber Attack Detection

This repository provides a clean implementation of the **proposed method** in `IEEE_ACCESS IRL/access_marked.tex`:

- State-based Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
- Reward learning from **normal traffic only**
- Threshold-based anomaly detection using reward percentile

This package intentionally excludes graph and table generation scripts.

## Method Summary

The model learns a reward function `R_theta(s)` for network-flow states `s`.

- Training objective (state-based MaxEnt IRL):
  - maximize likelihood of normal states under `P(s|R) ∝ exp(R_theta(s))`
  - implemented as minimizing:
    - `L = -mean(R_theta(s) - logsumexp(R_theta(batch)))`
- Detection rule:
  - compute threshold `tau` as `alpha`-th percentile of normal training rewards
  - predict **attack** if `R_theta(s_test) < tau`, otherwise **normal**

## Project Structure

```text
Github/
├── requirements.txt
├── README.md
├── train.py
├── predict.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    └── pipeline.py
```

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Default expected path:

`../Data/RT_IOT/RT_IOT2022.csv`

You can override this with `--data-path`.

## Train

```bash
python train.py --data-path "../Data/RT_IOT/RT_IOT2022.csv" --epochs 200 --batch-size 256 --threshold-percentile 5
```

Training outputs are stored in `artifacts/`:

- `reward_network.weights.h5`
- `normal_train_rewards.npy`
- `model_metadata.json`

## Inference / Evaluation

```bash
python predict.py --data-path "../Data/RT_IOT/RT_IOT2022.csv" --model-dir artifacts
```

The script reloads saved weights and threshold, then reports:

- accuracy
- precision
- recall
- f1
- roc_auc

## Notes for Reproducibility

- Seeds are fixed in code (`42`) for NumPy and TensorFlow.
- Preprocessing follows the paper implementation:
  - one-hot encoding for `proto` and `service`
  - standardization of numeric features (except `id.orig_p`, `id.resp_p`)
  - normal classes defined as:
    - `MQTT_Publish`, `Thing_Speak`, `Wipro_bulb`

## Publish to GitHub

1. Create a new repository on GitHub.
2. Upload the contents of this `Github/` folder.
3. Add a short repository description (example):
   - `State-based MaxEnt IRL for IoT network attack detection`
4. In GitHub repo settings:
   - enable `README` display (default)
   - optionally add a license (`MIT` or your preferred license)
5. If desired, add `RT_IOT2022.csv` locally but do not commit large/private data unless allowed.

## Citation

If you use this implementation, please cite your IEEE Access manuscript and reference the method section describing the state-based MaxEnt IRL formulation.

