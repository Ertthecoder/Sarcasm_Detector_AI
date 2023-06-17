import json
import pandas as pd
import numpy as np
import seaborn as sns
import fast_ml
from fast_ml.model_development import train_valid_test_split
import nltk
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, svm
from sklearn.metrics import f1_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
import re
import string
import matplotlib
import matplotlib.pyplot as plt
string.punctuation

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenize_data(dataframe):
    tokens = nltk.word_tokenize(dataframe)
    return [w for w in tokens if w.isalpha()]

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    output= [i for i in text if i not in stopwords]
    return output

def lemmatizer(text):
    wnlemmatizer = WordNetLemmatizer()
    lemm_text = [wnlemmatizer.lemmatize(word) for word in text]
    return lemm_text

def metric_results(algo):
    pred = algo.predict(X_valid)
    f1 = f1_score(pred, y_valid, average="weighted")
    recall = recall_score(pred, y_valid, average="weighted")
    print(f"{algo} f1 score: {f1}")
    print(f"{algo} recall score: {recall}")

    # Making the confusion matrix
    labels = ["Sarcastic", "Not Sarcastic"]
    cm = confusion_matrix(y_valid, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

if __name__ == '__main__':
    data = list(parse_data('./Sarcasm_Headlines_Dataset.json'))
    df = pd.DataFrame(data)
    print(df.head())
    # PREPROCESSING ----------------------------------------------------------------------------------------------------

    # Removing the link column from the df since we won't use it
    df = df.drop(columns=["article_link"])
    print(df.head(),"\n")

    """
    # Removing punctuation
    df['wo_punc'] = df['headline'].apply(lambda x: remove_punctuation(x))

    # Tokenization
    df['tokenized'] = df.apply(lambda x: tokenize_data(x['wo_punc']), axis=1)

    # Stop-word removal
    df['no_stopwords'] = df['tokenized'].apply(lambda x: remove_stopwords(x))

    # Lemmatization
    df['lemmatized'] = df['no_stopwords'].apply(lambda x: lemmatizer(x))
    """
    # Removing the punctuation & stop-word removal steps worked better in my case but I'm leaving them here for
    # different cases

    # Tokenization
    df['tokenized'] = df.apply(lambda x: tokenize_data(x['headline']), axis=1)

    # Lemmatization
    df['lemmatized'] = df['tokenized'].apply(lambda x: lemmatizer(x))

    print(df.head())

    # Vectorization with TF-IDF
    df['text'] = df['lemmatized'].apply(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df['text'])
    tfidf_vectors = vectorizer.transform(df['text'])
    tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=vectorizer.get_feature_names())
    final_df = pd.concat([df['is_sarcastic'], tfidf_df], axis=1)
    print(final_df.head())
    #print(tfidf_vectors.shape)

    # Splitting the df_t into train, test and validation sets
    df_l = df[['lemmatized', 'is_sarcastic']].copy()
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(final_df[:20000], target='is_sarcastic',
    train_size=0.8, test_size=0.1, valid_size=0.1, random_state=0)
    # Note(I had to trim the set a little bit above(line:105) due to hardware limitations)

    # Resetting indexes for a sorted set
    for d in [X_train, y_train, X_valid, y_valid, X_test, y_test]:
        d.reset_index(drop=True, inplace=True)

    # Splitted datasets' sizes
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_valid.shape, y_valid.shape)

    # APPLYING THE ALGORITHMS-------------------------------------------------------------------------------------------

    # The Naive Bayesian Analysis
    NBA = GaussianNB()
    NBA.fit(X_train, y_train)

    # Predicting validation set results
    metric_results(NBA)

    # Linear SVM
    SVM = SVC(kernel='linear')
    SVM.fit(X_train, y_train)

    # Predicting validation set results
    metric_results(SVM)

    
