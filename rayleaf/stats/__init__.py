import os

from pathlib import Path
from typing import Union


from.graphing import line_plot


"""
Raw client statistic keys
"""
CLIENT_ID_KEY = "client_id"
ROUND_NUMBER_KEY = "round_number"
NUM_CORRECT_KEY = "num_correct"
NUM_SAMPLES_KEY = "num_samples"
LOSS_KEY = "loss"


"""
Aggregate statistics keys
"""
AVERAGE_KEY = "average"
MEDIAN_KEY = "median"
def PERCENTILE_KEY(percentile: int) -> str:
    last_digit = percentile % 10
    if last_digit == 1:
        suffix = "st"
    elif last_digit == 2:
        suffix = "nd"
    elif last_digit == 3:
        suffix = "rd"
    else:
        suffix = "th"

    return f"{percentile}{suffix} percentile"

    
"""
Path constants
"""
def STATS_DIR(output_dir: Union[str, Path]) -> Path:
    stats_dir = Path(output_dir, "stats")
    if not stats_dir.is_dir():
        os.makedirs(stats_dir, exist_ok=True)
    
    return stats_dir


def STATS_CSV(stats_dir: Union[str, Path], stat_set: str, aggregate: bool) -> Path:
    csv_name = f"{stat_set}_stats.csv"
    if aggregate:
        csv_name = f"agg_{csv_name}"
    
    return Path(stats_dir, csv_name)


def GRAPHS_DIR(output_dir: Union[str, Path]) -> Path:
    stats_dir = Path(output_dir, "graphs")
    if not stats_dir.is_dir():
        os.makedirs(stats_dir, exist_ok=True)
    
    return stats_dir
