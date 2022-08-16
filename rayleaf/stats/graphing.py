from fileinput import filename
from pathlib import Path
from typing import Union


import numpy as np

from matplotlib import pyplot as plt


def line_plot(
    output_dir: Union[Path, str],
    title: str,
    x: Union[np.ndarray, list],
    xlabel: str,
    ylabel: str,
    marker: str,
    **kwargs
):
    file_name = title.lower().replace(" ", "_")
    graph_path = Path(output_dir, f"{file_name}.png")

    fig = plt.figure()

    for k, y in kwargs.items():
        plt.plot(x, y, label=k, marker=marker)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(graph_path)
    plt.close(fig)
