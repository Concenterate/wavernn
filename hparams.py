

class params:
    epochs = 10
    sample_rate = 22050
    nFft = 1024
    hop_size = 256
    constant_values = 0.
    num_mels = 64
    checkpoint_every = 100
    curr_index = 0
    raw_data_dirs = ["/home/pravesh/speech_dataset/speakers/3", ]
    train_data_dir = "training_data/"
    log_file_name = "log/wavernn.log/"
    model_file_name = "log/wavernn.model/"
    meta_data_file_name = "metadata.txt"
    sequence_units = 256
    prenet_outp = 256
    num_of_workers = 8
    bit_rate_in_power = 16
    scale_factor = 4
    model_type = "wavernn"
    seq_length = 1024
