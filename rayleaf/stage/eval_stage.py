import numpy as np


import rayleaf.metrics.writer as metrics_writer
import rayleaf.utils.logging_utils as logging_utils
from rayleaf.entities.server import Server


def eval_server(
    server: Server,
    client_num_samples,
    round: int,
    stat_writer_fn,
    use_val_set: bool
):
    print_stats(round, server, client_num_samples, stat_writer_fn, use_val_set)


def print_stats(num_round, server, num_samples, writer, use_val_set):
    train_stat_metrics = server.eval_model(set_to_use="train")
    print_metrics(train_stat_metrics, num_samples, prefix="train_")
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.eval_model(set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix="{}_".format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=""):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        logging_utils.log("%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g" \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.nanpercentile(ordered_metric, 10),
                 np.nanpercentile(ordered_metric, 50),
                 np.nanpercentile(ordered_metric, 90)))