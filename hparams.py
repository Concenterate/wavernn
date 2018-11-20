
"""

    This file contains all of the hyper parameters used by the wavernn model.

"""


class params:
    epochs = 10
    sample_rate = 22050  # 22KHz audio
    nFft = 1024
    hop_size = 256
    num_mels = 64
    checkpoint_every = 100
    input_data_dirs = ["/home/pravesh/speech_dataset/speakers/3", ]
    train_data_dir = "training_data"
    log_dir = "log/"

    # taken from tacotron2 project
    fmin = 125
    fmax = 7800
    max_db = 100

    # specific to preprocessing
    num_of_workers = 8
    bit_rate_in_power = 16
    scale_factor = int(hop_size/num_mels)

    # specific to rnn@896
    seq_cells = [128, 512]
    rnn_resolution = 256
    seq_length = 2048
    prenet_dropout = 0.2
