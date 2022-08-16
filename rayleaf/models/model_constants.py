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
    },
    "speech_commands.m5": {
        "num_classes": 35
    }
}
