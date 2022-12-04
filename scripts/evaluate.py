import argparse
import pickle
import numpy as np

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

from data.constants import ALL_LABELS_SORTED
from tuwnlpie.milestone2.utils import compute_metrics, get_test_dataset
from transformers import AutoModelForSequenceClassification
from transformers import Trainer


def evaluate_milestone1(test_data, saved_model):
    logger.info("Evaluating...")
    features, labels = read_preprocessed_features(test_data)
    model = pickle.load(open(saved_model, "rb"))
    x, y_true = get_xy(features, labels)
    y_pred = model.predict(x)
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    labels = y_true.columns

    for index in range(len(conf_matrix)):
        print(f"Confusion Matrix for Label: {labels[index]}:")
        print(conf_matrix[index])
        print("\n")

    return


def evaluate_milestone2(data_path, saved_model, split):
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(saved_model,num_labels=len(ALL_LABELS_SORTED),problem_type="multi_label_classification").to("cuda")
    
    logger.info("Loading data...")
    test_dataset = get_test_dataset(data_path)

    logger.info("Creating trainer...")
    # Create trainer with loaded model
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )
    
    logger.info("Evaluating...")
    evaluation = trainer.evaluate(test_dataset)

    print("Statistics:")
    print(f"Loss: {evaluation['eval_loss']}")
    print(f"Accuracy: {evaluation['eval_accuracy']}")
    print(f"F1-Score: {evaluation['eval_f1']}")
    print(f"ROC AUC: {evaluation['eval_roc_auc']}")

    logger.info("Predicting...")
    prediction = trainer.predict(test_dataset)
    predictions = (np.sign(prediction.predictions)+1)/2
    true_labels = test_dataset.labels.iloc[:,2:]
    conf_matrix = multilabel_confusion_matrix(true_labels, predictions)
    labels =  true_labels.columns
    for index in range(len(conf_matrix)):
        print(f"Confusion Matrix for Label: {labels[index]}:")
        print(conf_matrix[index])
        print("\n")
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
    args = get_args()

    test_data = args.test_data
    model = args.saved_model
    split = args.split
    milestone = args.milestone

    if milestone == 1:
        evaluate_milestone1(test_data, model)
    elif milestone == 2:
        evaluate_milestone2(test_data, model, split=split)
