from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import Generator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--encoder", type=str, default="vits16_800ep")

    parser.add_argument("--n_labels_end", type=int, default=10_000)

    parser.add_argument("--inverse_l2_coef", type=float, default=1)
    parser.add_argument("--n_optim_steps_max", type=int, default=1_000)

    return parser.parse_args()


def sample_n_per_class(labels: np.ndarray, n_per_class: int, rng: Generator) -> np.ndarray:
    sampled_inds = []

    for _class in np.unique(labels):
        class_inds = np.flatnonzero(labels == _class)
        sampled_inds += [rng.choice(class_inds, size=n_per_class, replace=False)]

    sampled_inds = np.concatenate(sampled_inds)
    sampled_inds = rng.permutation(sampled_inds)

    return sampled_inds


def compute_mean_and_std(
    x: np.ndarray, axis: int = 0, keepdims: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    To avoid dividing by zero we set the standard deviation to one if it is less than epsilon.
    """
    mean = np.mean(x, axis=axis, keepdims=keepdims)
    std = np.std(x, axis=axis, keepdims=keepdims)

    eps = np.finfo(x.dtype).eps

    if isinstance(std, np.ndarray):
        std[std < eps] = 1
    elif std < eps:
        std = 1

    return mean, std


def save_table(path: Union[Path, str], table: pd.DataFrame, formatting: dict) -> None:
    for key in formatting:
        if key in table:
            table[key] = table[key].apply(formatting[key])

    table.to_csv(path, index=False)


def main(cfg: Namespace) -> None:
    rng = np.random.default_rng(cfg.seed)

    data_dir = Path(cfg.data_dir) / cfg.dataset
    results_dir = Path(cfg.results_dir) / cfg.encoder
    results_dir.mkdir(parents=True, exist_ok=True)

    train_inputs = np.load(data_dir / f"embeddings_{cfg.encoder}_train.npy")
    train_labels = np.load(data_dir / "labels_train.npy")

    test_inputs = np.load(data_dir / f"embeddings_{cfg.encoder}_val.npy")
    test_labels = np.load(data_dir / "labels_val.npy")

    mean, std = compute_mean_and_std(train_inputs)
    train_inputs = (train_inputs - mean) / std
    test_inputs = (test_inputs - mean) / std

    n_per_class = 1
    n_labels, accs, logliks = [], [], []

    while True:
        train_inds = sample_n_per_class(train_labels, n_per_class, rng)

        model = LogisticRegression(C=cfg.inverse_l2_coef, max_iter=cfg.n_optim_steps_max)
        model.fit(train_inputs[train_inds], train_labels[train_inds])

        probs = model.predict_proba(test_inputs)
        acc = model.score(test_inputs, test_labels)
        loglik = -log_loss(test_labels, probs)

        n_labels += [len(train_inds)]
        accs += [acc]
        logliks += [loglik]

        if len(train_inds) >= cfg.n_labels_end:
            break
        else:
            n_per_class += 1

    test_log = pd.DataFrame({"n_labels": n_labels, "test_acc": accs, "test_loglik": logliks})
    formatting = {
        "n_labels": "{:03}".format,
        "test_acc": "{:.4f}".format,
        "test_loglik": "{:.4f}".format,
    }
    save_table(results_dir / f"testing_seed{cfg.seed:02}.csv", test_log, formatting)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
