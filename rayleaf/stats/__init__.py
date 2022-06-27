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
def PERCENTILE_KEY(percentile: int):
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
