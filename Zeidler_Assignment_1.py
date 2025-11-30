import argparse

from patient_retriever import PatientRetriever
import sys

def main():
    args = check_args()
    patient_retriever = PatientRetriever(args[0],args[1])
    patient_retriever.find_top_5()
    patient_retriever.similar_patients_to_json()
    patient_retriever.metrics_to_json()
    patient_retriever.plot_precision_recall()


def check_args():
    if (args := sys.argv[1:]) is None:
        raise TypeError("Please provide arguments")
    if type(args[0]) != str:
        raise TypeError("First argument must be a string")
    if len(args) > 1 and type(args[1]) != int:
        raise TypeError("Second argument must be a integer")
    return args


main()