import argparse
import pickle
from os import getcwd, chdir, path
from sklearn.neural_network import MLPClassifier

from tuwnlpie import logger
from tuwnlpie.milestone1.utils import (
    calculate_tp_fp_fn,
    read_docs_from_csv,
    split_train_dev_test,
    read_preprocessed_features,
    get_xy
)
from tuwnlpie.milestone2.model import train_model


def train_milestone2(path_to_data, save=False, save_path=None):
    logger.info("Loading data...")

    path_to_data=path_to_data
    train_model(path_to_data,save_path)

    return


def train_on_preprocessed_milestone_1(path_to_data, save=False, save_path=None):


    logger.info("Loading data...")
    features, labels = read_preprocessed_features(path_to_data)
    x, y = get_xy(features, labels)

    logger.info("Loading model, random state fixed...")
    #model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    model= MLPClassifier(random_state=1, max_iter=290)

    logger.info("Training...")
    model.fit(x, y)
    if save:
        pickle.dump(model, open(save_path, 'wb'))
        logger.info(f"Saved model to {save_path}")
    return


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--train-data", type=str, required=True, help="Path to folder in which are training data stored"
    )
    parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save model"
    )
    parser.add_argument(
        "-sp", "--save-path", default=None, type=str, help="Path to save model"
    )
    parser.add_argument(
        "-m", "--milestone", type=int, choices=[1, 2], help="Milestone to train"
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    train_data = args.train_data
    model_save = args.save
    model_save_path = args.save_path
    milestone = args.milestone

    if milestone == 1:
        train_on_preprocessed_milestone_1(train_data, save=model_save, save_path=model_save_path)
    elif milestone == 2:
        train_milestone2(train_data, save=model_save, save_path=model_save_path)
