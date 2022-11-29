
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from data.constants import ALL_LABELS_SORTED
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,EvalPrediction
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


# Path to csv files
WD = '../'
data_path = WD + 'data/'
save_path=WD +'data/milestone2'
train_df = pd.read_csv(data_path + 'train_data.csv')
valid_df = pd.read_csv(data_path + 'valid_data.csv')
test_df = pd.read_csv(data_path + 'test_data.csv')

def preprocess_labels(df: pd.DataFrame):
    labels =  df.iloc[:, [0, 2]].copy()
    labels.Labels = labels.Labels.apply(ast.literal_eval)
    mlb = MultiLabelBinarizer(classes=list(ALL_LABELS_SORTED.keys()))
    labels_transformed = mlb.fit_transform(labels['Labels'])
    labels[mlb.classes_] = labels_transformed
    return  labels

train_labels=preprocess_labels(train_df)
valid_labels=preprocess_labels(valid_df)

train_sent=train_df['Text'].tolist()
valid_sent=valid_df["Text"].tolist()

model_ckpt = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt,problem_type="multi_label_classification")

train_encodings = tokenizer(train_sent,truncation=True,padding=True,return_tensors='pt')
valid_encodings = tokenizer(valid_sent, truncation=True, padding=True, return_tensors='pt')

class BRISEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.FloatTensor(self.labels.iloc[idx,2:])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BRISEDataset(train_encodings, train_labels)
valid_dataset=BRISEDataset(valid_encodings,valid_labels)


model = AutoModelForSequenceClassification.from_pretrained(model_ckpt,num_labels=len(ALL_LABELS_SORTED),problem_type="multi_label_classification").to("cuda")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                  num_train_epochs=5,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  optim="adamw_torch",
                                  metric_for_best_model = "f1")


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

from transformers import  Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()
