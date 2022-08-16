from pathlib import Path
from typing import Union


import pandas as pd


import rayleaf.stats as stats


def graph(output_dir: Union[Path, str], eval_set: str):
    stats_dir = stats.STATS_DIR(output_dir)

    agg_train_stats = pd.read_csv(stats.STATS_CSV(stats_dir, "train", aggregate=True))
    agg_eval_stats = pd.read_csv(stats.STATS_CSV(stats_dir, eval_set, aggregate=True))

    rounds = agg_train_stats[stats.ROUND_NUMBER_KEY].values
    train_percentiles = _collect_percentile_cols(agg_train_stats, dataset="train")
    eval_percentiles = _collect_percentile_cols(agg_eval_stats, dataset=eval_set)

    graphs_dir = stats.GRAPHS_DIR(output_dir)
    stats.line_plot(
        output_dir=graphs_dir,
        title="Client Train Accuracies",
        x=rounds,
        xlabel="Round number",
        ylabel="Accuracy",
        marker=".",
        **train_percentiles
    )

    stats.line_plot(
        output_dir=graphs_dir,
        title=f"Client {eval_set.title()} Accuracies",
        x=rounds,
        xlabel="Round number",
        ylabel="Accuracy",
        marker=".",
        **eval_percentiles
    )

    averages = {
        "train": agg_train_stats[stats.AVERAGE_KEY],
        eval_set: agg_eval_stats[stats.AVERAGE_KEY]
    }
    stats.line_plot(
        output_dir=graphs_dir,
        title=f"Average Train vs {eval_set.title()} Accuracies",
        x=rounds,
        xlabel="Round number",
        ylabel="Accuracy",
        marker=".",
        **averages
    )


def _collect_percentile_cols(stats_df: pd.DataFrame, dataset: str):
    cols = {}
    for col in stats_df.columns:
        if col != stats.ROUND_NUMBER_KEY and col != stats.AVERAGE_KEY:
            cols[col] = stats_df[col].values
    
    return cols
