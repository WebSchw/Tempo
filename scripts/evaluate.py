import argparse
import pickle

from sklearn.metrics import multilabel_confusion_matrix

from tuwnlpie import logger
from sklearn.neural_network import MLPClassifier
from tuwnlpie.milestone1.utils import (
    calculate_tp_fp_fn,
    read_docs_from_csv,
    split_train_dev_test,
    read_preprocessed_features,
    get_xy
)

from tuwnlpie.milestone2.model import BoWClassifier
from tuwnlpie.milestone2.utils import IMDBDataset, Trainer


def evaluate_milestone1(test_data, saved_model, split=False):
    features, labels = read_preprocessed_features(test_data)
    model=pickle.load(open(saved_model,"rb"))
    x, y_true = get_xy(features, labels)
    y_pred=model.predict(x)
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    labels=y_true.columns

    for index in range(len(conf_matrix)):
        print(f"for label {labels[index]} Confusion matrix:")
        print(conf_matrix[0])
        print("\n")





    return


def evaluate_milestone2(test_data, saved_model, split=False):
    logger.info("Loading data...")
    dataset = IMDBDataset(test_data)
    model = BoWClassifier(dataset.OUT_DIM, dataset.VOCAB_SIZE)
    model.load_model(saved_model)
    trainer = Trainer(dataset=dataset, model=model)

    logger.info("Evaluating...")
    test_loss, test_prec, test_rec, test_fscore = trainer.evaluate(
        dataset.test_iterator
    )

    print("Statistics:")
    print(f"Loss: {test_loss}")
    print(f"Precision: {test_prec}")
    print(f"Recall: {test_rec}")
    print(f"F1-Score: {test_fscore}")

    return


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--test-data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "-sm", "--saved-model", type=str, required=True, help="Path to saved model"
    )
    parser.add_argument(
        "-sp", "--split", default=False, action="store_true", help="Split data"
    )
    parser.add_argument(
        "-m", "--milestone", type=int, choices=[1, 2], help="Milestone to evaluate"
    )

    return parser.parse_args()


if "__main__" == __name__:
    #args = get_args()
    test_data=r"data/input/test_data.csv"
    model=r"data\models\rf"
    split=None
    milestone=1

    # test_data = args.test_data
    # model = args.saved_model
    # split = args.split
    # milestone = args.milestone

    if milestone == 1:
        evaluate_milestone1(test_data, model, split=split)
    elif milestone == 2:
        evaluate_milestone2(test_data, model, split=split)
