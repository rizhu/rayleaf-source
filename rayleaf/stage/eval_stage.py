import numpy as np
import pandas as pd


import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils
from rayleaf.entities.server import Server
import rayleaf.stats as stats


def eval_server(
    server: Server,
    num_round: int,
    eval_set: bool
):
    train_stats, eval_stats = _eval_clients(server, num_round, eval_set=eval_set)


    agg_train_stats, agg_eval_stats = _compute_stats(train_stats), _compute_stats(eval_stats)
    _print_aggregate_stats(agg_train_stats, header=f"training accuracy: round {num_round}".title())
    _print_aggregate_stats(agg_eval_stats, header=f"{eval_set} accuracy: round {num_round}".title())
    agg_train_stats[stats.ROUND_NUMBER_KEY], agg_eval_stats[stats.ROUND_NUMBER_KEY] = num_round, num_round


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

    return aggregate_stats


def _print_aggregate_stats(stats: dict, header: str):
    logging_utils.log(utils.EVAL_STR.format(header))

    stats_df = pd.DataFrame([stats])
    logging_utils.log_df(stats_df)
