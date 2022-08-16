import pandas as pd

from tabulate import tabulate


logging_file = None


def log(msg: str = "") -> None:
    global logging_file

    print(msg, file=logging_file)
    print(msg)


def log_df(df: pd.DataFrame):
    log(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


def shutdown_log() -> None:
    global logging_file
    
    if logging_file is not None:
        logging_file.close()
    logging_file = None
