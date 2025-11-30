from patient_retriever import PatientRetriever

def main():
    patient_retriever = PatientRetriever(directory_path="/home/user/PycharmProjects/HI744_Programming_Assignment_1/",nrows=1000)
    patient_retriever.find_top_5()
main()