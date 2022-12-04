from transformers import AutoTokenizer, EvalPrediction
from tuwnlpie.milestone2.model import BRISEDataset, preprocess_labels
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result


def get_test_dataset(data_path):
    test_df = pd.read_csv(data_path + 'test_data.csv')
    test_labels = preprocess_labels(test_df)
    test_sent = test_df["Text"].tolist()

    model_ckpt = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt,problem_type="multi_label_classification")
    
    test_encodings = tokenizer(test_sent, truncation=True, padding=True, return_tensors='pt')
    test_dataset = BRISEDataset(test_encodings,test_labels)
    
    return test_dataset