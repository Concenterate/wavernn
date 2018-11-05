from librosa.core import load as librosa_import
from librosa.feature import melspectrogram
import numpy as np
from glob import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
from wavernn.hparams import params as pm

scale_factor = int(2**(pm.bit_rate_in_power-1))
split_factor = int(2**(pm.bit_rate_in_power/2))
second_split = int(split_factor/2)


def postprocess_data(linear, mel, file_name):

    num_of_frames = mel.shape[0]
    # scale the signal, from [-1 to 1] to [0 to bit_Rate]
    bit_rate_signal = scale_factor*(linear+1)
    # take first_half of bits as coarse, while remaining as fine
    coarse_of_signal, fine_of_signal = np.divmod(bit_rate_signal, split_factor)

    # merge all linear data in one stack and reshape wrt to mel data
    linear_data = np.stack([coarse_of_signal, fine_of_signal], axis=0)

    np.savez_compressed(file_name, linear=linear_data, mel=mel)

    return None


def process_individual_file(input_file_uri, index):

    data = [librosa_import(uri)[0] for uri in input_file_uri][0]

    mfcc = melspectrogram(data, n_fft=pm.nFft,
                          hop_length=pm.hop_size,
                          n_mels=pm.num_mels).T

    mfcc_shape = mfcc.shape
    num_elem_in_mfcc = mfcc_shape[0]*mfcc_shape[1]
    pad_val = pm.scale_factor*num_elem_in_mfcc - data.shape[0]

    data = np.pad(data, [0, pad_val], mode="constant")

    assert data.shape[0] % num_elem_in_mfcc == 0

    output_file_uri = os.path.join(
        training_data_folder, "sample_{}.npz".format(index))
    postprocess_data(data, mfcc, output_file_uri)

    return "{}|{}|{}\n".format(output_file_uri, mfcc_shape[0], len(data))


def preprocess(input_folders):
    concurrent_executor = ThreadPoolExecutor(max_workers=pm.num_of_workers)

    source_files = [glob(source_folder+"/*.wav")
                    for source_folder in input_folders]

    number_of_files = list(map(len, source_files))
    min_constraint = np.min(number_of_files)

    shuffled_entries = []
    for index, each_file_len in enumerate(number_of_files):
        np.random.shuffle(source_files[index])
        shuffled_entries.append(source_files[index][:min_constraint])
    shuffled_array = np.asarray(shuffled_entries).T
    indices = np.arange(len(shuffled_array))

    meta_data = open(training_data_folder +
                     "/{}".format(pm.meta_data_file_name), 'w')
    for e in tqdm(concurrent_executor.map(process_individual_file, shuffled_array, indices)):
        meta_data.write(e)
    meta_data.flush()


def run_preprocess():

    global training_data_folder
    training_data_folder = pm.train_data_dir
    os.makedirs(training_data_folder, exist_ok=True)

    input_folders = pm.raw_data_dirs

    preprocess(input_folders)


def main():
    run_preprocess()


if __name__ == '__main__':
    main()
