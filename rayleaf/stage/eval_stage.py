import os
from pathlib import Path


import numpy as np
import pandas as pd


import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils
from rayleaf.entities.server import Server
import rayleaf.stats as stats


def eval_server(
    server: Server,
    num_round: int,
    eval_set: bool,
    output_dir: str
):
    output_dir = Path(output_dir)
    stats_dir = stats.STATS_DIR(output_dir=output_dir)

    train_stats, eval_stats = _eval_clients(server, num_round, eval_set=eval_set)

    train_stats.to_csv(stats.STATS_CSV(stats_dir, "train", aggregate=False), mode="a+", index=False, header=True)
    eval_stats.to_csv(stats.STATS_CSV(stats_dir, eval_set, aggregate=False), mode="a+", index=False, header=True)

    agg_train_stats, agg_eval_stats = _compute_stats(train_stats), _compute_stats(eval_stats)
    _print_aggregate_stats(agg_train_stats, header=f"training accuracy: round {num_round}".title())
    _print_aggregate_stats(agg_eval_stats, header=f"{eval_set} accuracy: round {num_round}".title())
    agg_train_stats[stats.ROUND_NUMBER_KEY], agg_eval_stats[stats.ROUND_NUMBER_KEY] = num_round, num_round

    agg_train_stats.to_csv(stats.STATS_CSV(stats_dir, "train", aggregate=True), mode="a+", index=False, header=True)
    agg_eval_stats.to_csv(stats.STATS_CSV(stats_dir, eval_set, aggregate=True), mode="a+", index=False, header=True)


def _eval_clients(
    server: Server,
    num_round: int,
    eval_set: str
):
    train_stats = server.eval_model(set_to_use="train")
    eval_stats = server.eval_model(set_to_use=eval_set)

    train_stats, eval_stats = pd.DataFrame(train_stats), pd.DataFrame(eval_stats)
    train_stats[stats.ROUND_NUMBER_KEY], eval_stats[stats.ROUND_NUMBER_KEY] = num_round, num_round

    return train_stats, eval_stats


def _compute_stats(
    stats_df: pd.DataFrame
):
    num_correct = stats_df[stats.NUM_CORRECT_KEY].values
    num_samples = stats_df[stats.NUM_SAMPLES_KEY].values

    accuracy = num_correct / num_samples

    avg = np.average(accuracy, weights=num_samples)
    med = np.median(accuracy)
    percentiles = {
        10: np.nanpercentile(accuracy, 10),
        90: np.nanpercentile(accuracy, 90)
    }

    aggregate_stats = {
        stats.AVERAGE_KEY: avg,
        stats.MEDIAN_KEY: med
    }
    for percentile, value in percentiles.items():
        aggregate_stats[stats.PERCENTILE_KEY(percentile)] = value

    return pd.DataFrame([aggregate_stats])


def _print_aggregate_stats(stats: pd.DataFrame, header: str):
    logging_utils.log(utils.EVAL_STR.format(header))

    logging_utils.log_df(stats)
