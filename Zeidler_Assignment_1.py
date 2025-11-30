import argparse

from patient_retriever import PatientRetriever
import sys

def main():
    nrows = 10000
    path = check_args()
    patient_retriever = PatientRetriever(path,nrows)
    patient_retriever.find_top_5()
    patient_retriever.similar_patients_to_json()
    patient_retriever.metrics_to_json()
    patient_retriever.plot_precision_recall()


def check_args():
    if (len(args := sys.argv[1:])) < 1:
        raise TypeError("Please provide arguments")
    if len(args) != 1:
        raise TypeError("Please provide one argument")
    if type(args[0]) != str:
        raise TypeError("Argument must be a string")
    return args[0]

if __name__ == "__main__":
    main()