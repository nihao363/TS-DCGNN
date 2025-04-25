import argparse

from tqdm import tqdm
import pickle
import os
import json
import pandas as pd
import numpy as np

import utils

from sentence_transformers import SentenceTransformer

# Initialize EEG model for feature embeddings
sbert_model = SentenceTransformer("paraphrase-distilroberta-base-v1")

log = utils.get_logger()


# Define a sample class for EEG emotion recognition
class EEGEmotionSample:
    def __init__(self, vid, label, eeg_data, sentence):
        self.vid = vid
        self.label = label
        self.eeg_data = eeg_data  # EEG data corresponding to the video/sample
        self.sentence = sentence
        self.sbert_sentence_embeddings = sbert_model.encode(sentence)  # Sentence embeddings for potential use


def get_eeg_data():
    utils.set_seed(args.seed)

    if args.dataset == "seed_eeg":
        # Replace with the path to the EEG dataset for seed (or your desired dataset)
        (
            video_ids,
            video_labels,
            eeg_data,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/seed_eeg/SEED_EEG_features.pkl", "rb"), encoding="latin1"
        )

    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            EEGEmotionSample(
                vid,
                video_labels[vid],
                eeg_data[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            EEGEmotionSample(
                vid,
                video_labels[vid],
                eeg_data[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            EEGEmotionSample(
                vid,
                video_labels[vid],
                eeg_data[vid],
                video_sentence[vid],
            )
        )

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def main(args):
    if args.dataset == "seed_eeg":
        train, dev, test = get_eeg_data()
        data = {"train": train, "dev": dev, "test": test}
        utils.save_pkl(data, "./data/seed_eeg/data_seed_eeg.pkl")

    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    parser.add_argument(
        "--dataset",
        type=str,
        default="seed_eeg",  # Change this if you are using a different EEG dataset
        help="Dataset name: seed_eeg, etc.",
    )

    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset directory"
    )
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    args = parser.parse_args()

    main(args)
