import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import os
import pandas as pd
import utils



class PatientRetriever:
    def __init__(self, directory_path,nrows=None):
        self.fname = 'PMC-Patients.csv'
        self.dataset = PatientRetriever.load_dataset(proj_directory = directory_path,fname = self.fname,nrows = nrows)

    @staticmethod
    def safe_read_csv(proj_directory:str, fname: str, nrows):
        try:
            filepath = os.path.join(proj_directory, fname)
            return pd.read_csv(filepath_or_buffer=filepath, nrows=nrows)
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    @staticmethod
    def load_dataset(proj_directory:str,fname: str, nrows):
        data = PatientRetriever.safe_read_csv(proj_directory,fname, nrows)
        data['tokens'] = data['patient'].apply(lambda text: utils.pre_process(text))
        print("Data loaded")
        return data

