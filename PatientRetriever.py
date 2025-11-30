import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import os
import pandas as pd
import utils




class PatientRetriever:
    def __init__(self, directory_path,nrows=None):
        self.fname = 'PMC-Patients.csv'
        self.dataset = utils.load_dataset(proj_directory = directory_path,fname = self.fname,nrows = nrows)


