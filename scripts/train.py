import argparse
import pickle

from sklearn.neural_network import MLPClassifier

from tuwnlpie import logger
from tuwnlpie.milestone1.model import SimpleNBClassifier
from tuwnlpie.milestone1.utils import (
    calculate_tp_fp_fn,
    read_docs_from_csv,
    split_train_dev_test,
    read_preprocessed_features,
    get_xy
)
from sklearn.ensemble import RandomForestClassifier

from tuwnlpie.milestone2.model import BoWClassifier
from tuwnlpie.milestone2.utils import IMDBDataset, Trainer


def train_milestone2(train_data, save=False, save_path=None):
    logger.info("Loading data...")
    dataset = IMDBDataset(train_data)
    model = BoWClassifier(dataset.OUT_DIM, dataset.VOCAB_SIZE)
    trainer = Trainer(dataset=dataset, model=model)

    logger.info("Training...")
    trainer.training_loop(dataset.train_iterator, dataset.valid_iterator)

    if save:
        model.save_model(save_path)

    return


def train_on_preprocessed_milestone_1(path_to_data, save=False, save_path=None):
    #model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    model=classifier_deep = MLPClassifier(random_state=1, max_iter=290)
    features, labels = read_preprocessed_features(path_to_data)
    x, y = get_xy(features, labels)
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
