import os
import pandas as pd
import utils
from doc2vec_ranker import Doc2VecRanker
import nltk




class PatientRetriever:
    def __init__(self, directory_path,nrows=None):
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        self.fname = 'PMC-Patients.csv'
        self.dataset = PatientRetriever.load_dataset(proj_directory = directory_path,fname = self.fname,nrows = nrows)

    @staticmethod
    def safe_read_csv(proj_directory: str, fname: str, nrows):
        try:
            filepath = os.path.join(proj_directory, fname)
            return pd.read_csv(filepath_or_buffer=filepath, nrows=nrows)
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    @staticmethod
    def load_dataset(proj_directory: str, fname: str, nrows):
        data = PatientRetriever.safe_read_csv(proj_directory, fname, nrows)
        data['tokens'] = data['patient'].apply(lambda text: utils.pre_process(text))
        print("Data loaded")
        return data

    def find_top_5(self):
        doc2vec = Doc2VecRanker(self.dataset)
        doc2vec.top_5("doc2vec_top5")
        print(self.dataset.head())
        return self.dataset

