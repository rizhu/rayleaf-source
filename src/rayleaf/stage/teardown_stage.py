import os

from datetime import datetime
from pathlib import Path


import rayleaf.utils as utils
import rayleaf.utils.logging_utils as logging_utils


def teardown(
    save_model: bool,
    output_dir: str,
    dataset: str,
    model,
    server,
    start_time
):
    if save_model:
        # Save server model
        ckpt_path = Path(output_dir, "checkpoints", dataset)
        if not ckpt_path.is_dir():
            os.makedirs(ckpt_path, exist_ok=True)
        save_path = Path(ckpt_path, f"{model}.ckpt")
        save_path = server.save_model(save_path)
        logging_utils.log(f"Model saved in path: {save_path}")
    
    end_time = datetime.now()
    logging_utils.log(f"Total Experiment time: {end_time - start_time}")

    logging_utils.shutdown_log()
