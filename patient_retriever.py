import os
import pandas as pd
import utils
from doc2vec_ranker import Doc2VecRanker
import nltk
from tfidf_ranker import TFIDFRanker
import matplotlib.pyplot as plt
import numpy as np
import ast



class PatientRetriever:
    def __init__(self, directory_path,nrows=None):
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        self.column_names = ['similar_patients_tfidf', 'similar_patients_doc2vec']
        self.fname = 'PMC-Patients.csv'
        self.dataset = PatientRetriever._load_dataset(proj_directory = directory_path, fname = self.fname, nrows = nrows)

    @staticmethod
    def _safe_read_csv(proj_directory: str, fname: str, nrows):
        try:
            filepath = os.path.join(proj_directory, fname)
            return pd.read_csv(filepath_or_buffer=filepath, nrows=nrows)
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    @staticmethod
    def _load_dataset(proj_directory: str, fname: str, nrows):
        data = PatientRetriever._safe_read_csv(proj_directory, fname, nrows)
        data['tokens'] = data['patient'].apply(lambda text: utils.pre_process(text))
        print("Data loaded")
        return data

    def find_top_5(self):
        tfidf = TFIDFRanker(self.dataset)
        tfidf.top_5(self.column_names[0])
        doc2vec = Doc2VecRanker(self.dataset)
        doc2vec.top_5(self.column_names[1])
        self.similar_patients_to_json()
        return self.dataset

    def similar_patients_to_json(self):
        print("Creating similar patients")
        similar_patients = self.dataset[['patient_uid', self.column_names[0], self.column_names[1]]]
        similar_patients.to_json('similar_patients.json', orient='records')
        print("Similar patients created")

    def metrics_to_json(self):
        print("Calculating metrics")
        self.dataset['similar_patients'] = self.dataset['similar_patients'].apply(lambda x: PatientRetriever._string_to_list(x))
        self.dataset['doc2vec_precision'], self.dataset['doc2vec_recall'] = zip(
            *self.dataset.apply(lambda row: PatientRetriever._calc_precision_recall(row['similar_patients'], row[self.column_names[1]]),
                                axis=1))
        self.dataset['tfidf_precision'], self.dataset['tfidf_recall'] = zip(
            *self.dataset.apply(lambda row: PatientRetriever._calc_precision_recall(row['similar_patients'], row[self.column_names[0]]),
                                axis=1))
        metrics = self.dataset[['patient_uid', 'doc2vec_precision', 'doc2vec_recall', 'tfidf_precision', 'tfidf_recall']]
        metrics.to_json('metrics.json', orient='records')
        print("Metrics calculated")
        print(self.dataset.head())

    @staticmethod
    def _string_to_list(text):
        if type(text) == list:
            return text
        else:
            return list(ast.literal_eval(text).keys())

    @staticmethod
    def _calc_precision_recall(y_true, y_pred):
        true_positives = len([patient for patient in y_pred if patient in y_true])
        false_positives = len(y_pred) - true_positives
        false_negatives = len(y_true) - true_positives
        precision = true_positives / (true_positives + false_positives)
        recall = 0 if true_positives + false_negatives == 0 else true_positives / (true_positives + false_negatives)
        return precision, recall

    def plot_precision_recall(self):
        if 'doc2vec_precision' not in self.dataset.columns:
            raise ValueError("Metrics must be calculated before plotting")
        categories = np.array(["doc2vec_precision", "tfidf_precision", "doc2vec_recall", "tfidf_recall"])
        doc2vec_avg_precision = self.dataset['doc2vec_precision'].mean()
        doc2vec_avg_rec = self.dataset['doc2vec_recall'].mean()
        tfidf_avg_precision = self.dataset['tfidf_precision'].mean()
        tfidf_avg_rec = self.dataset['tfidf_recall'].mean()
        values = np.array([doc2vec_avg_precision, doc2vec_avg_rec, tfidf_avg_precision, tfidf_avg_rec])
        plt.bar(categories, values, color=['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'])
        plt.xlabel("Document Representation")
        plt.ylabel("Score")
        plt.title("Average Precision and Recall @5 for Doc2Vec and TF-IDF")
        plt.savefig('average_precision_recall.png')

