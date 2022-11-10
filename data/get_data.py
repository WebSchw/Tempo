import argparse
import requests
import nltk
import stanza
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from data.constants import ALL_LABELS_SORTED


def download():
    csv_urls = [
        "https://raw.githubusercontent.com/recski/brise-plandok/main/brise_plandok/baselines/input/test_data.csv",
        "https://raw.githubusercontent.com/recski/brise-plandok/main/brise_plandok/baselines/input/train_data.csv",
        "https://raw.githubusercontent.com/recski/brise-plandok/main/brise_plandok/baselines/input/valid_data.csv"
    ]
    for url in csv_urls:
        r = requests.get(url, allow_redirects=True)
        set = url.split("/")[-1].split("_")[0]
        open(f'{set}_data.csv', 'wb').write(r.content)


def preprocess():
    nltk.download('stopwords')
    german_stop_words = nltk.corpus.stopwords.words('german')
    stanza.download('de')
    nlp = stanza.Pipeline(processors="tokenize,mwt,lemma", lang="de", use_gpu=True)
    train_df = pd.read_csv('train_data.csv')
    valid_df = pd.read_csv('valid_data.csv')
    test_df = pd.read_csv('test_data.csv')

    def tokenize(input: str):
        doc = nlp(input)
        return [word.lemma for sent in doc.sentences for word in sent.words if word.lemma not in german_stop_words]

    def preprocess_features(df: pd.DataFrame, type_of_data: str, vocab=None):
        vectorizer = CountVectorizer(tokenizer=tokenize, vocabulary=vocab, lowercase=False)
        features = df.iloc[:, [0, 1]].copy()
        vectors = vectorizer.fit_transform(features.Text).toarray()
        transformed_features = pd.DataFrame(vectors, columns=vectorizer.get_feature_names_out())
        features = features.join(transformed_features)
        features.to_csv((type_of_data + '_features.csv'), index=False)
        return features.columns[2:].tolist()

    def preprocess_labels(df: pd.DataFrame, type_of_data: str):
        labels = df.iloc[:, [0, 2]].copy()
        labels.Labels = labels.Labels.apply(ast.literal_eval)
        mlb = MultiLabelBinarizer(classes=list(ALL_LABELS_SORTED.keys()))
        labels_transformed = mlb.fit_transform(labels['Labels'])
        labels[mlb.classes_] = labels_transformed
        labels.to_csv((type_of_data + '_labels.csv'), index=False)

    def preprocess_df(df: pd.DataFrame, type_of_data: str, vocab=None):
        vocab = preprocess_features(df, type_of_data, vocab)
        preprocess_labels(df, type_of_data)
        return vocab

    vocab = preprocess_df(train_df, 'train')
    preprocess_df(valid_df, 'valid', vocab)
    preprocess_df(test_df, 'test', vocab)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d", "--download-data", default=False, action="store_true", help="download data"
    )
    parser.add_argument(
        "-dp", "--download-and-preprocess-data", default=False, action="store_true", help="download and preprocess data"
    )
    parser.add_argument(
        "-p", "--preprocess-data", default=False, action="store_true", help="preprocess data"
    )

    return parser.parse_args()


def get_data(do_preprocess):
    if not do_preprocess:
        download()
    else:
        preprocess()

if "__main__" == __name__:
    args = get_args()
    if not args.download_and_preprocess_data and not args.download_data and not args.preprocess_data:
        download()
        preprocess()
        exit()
    if args.download_and_preprocess_data:
        download()
        preprocess()
        exit()
    if args.download_data:
        download()
    if args.preprocess_data:
        preprocess()



