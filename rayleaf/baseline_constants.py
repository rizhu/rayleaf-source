"""Configuration file for common models/experiments"""

MAIN_PARAMS = { 
    "sent140": {
        "small": (10, 2, 2),
        "medium": (16, 2, 2),
        "large": (24, 2, 2)
        },
    "femnist": {
        "small": (30, 10, 2),
        "medium": (100, 10, 2),
        "large": (400, 20, 2)
        },
    "shakespeare": {
        "small": (6, 2, 2),
        "medium": (8, 2, 2),
        "large": (20, 1, 2)
        },
    "celeba": {
        "small": (30, 10, 2),
        "medium": (100, 10, 2),
        "large": (400, 20, 2)
        },
    "synthetic": {
        "small": (6, 2, 2),
        "medium": (8, 2, 2),
        "large": (20, 1, 2)
        },
    "reddit": {
        "small": (6, 2, 2),
        "medium": (8, 2, 2),
        "large": (20, 1, 2)
        },
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_SETTINGS = {
    "sent140.bag_dnn": {
        "num_classes": 2
    },
    "sent140.stacked_lstm": {
        "seq_len": 25,
        "num_classes": 2,
        "num_hidden": 100
    },
    "sent140.bag_log_reg": {
        "num_classes": 2
    },
    "femnist.cnn": {
        "num_classes": 62
    },
    "shakespeare.stacked_lstm": {
        "seq_len": 80,
        "num_classes": 80,
        "num_hidden": 256
    },
    "celeba.cnn": {
        "num_classes": 2
    },
    "synthetic.log_reg": {
        "num_classes": 5,
        "input_dim": 60
    },
    "reddit.stacked_lstm": {
        "seq_len": 10,
        "num_hidden": 256,
        "num_layers": 2
    }
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = "accuracy"
BYTES_WRITTEN_KEY = "bytes_written"
BYTES_READ_KEY = "bytes_read"
LOCAL_COMPUTATIONS_KEY = "local_computations"
NUM_ROUND_KEY = "round_number"
NUM_SAMPLES_KEY = "num_samples"
CLIENT_ID_KEY = "client_id"
