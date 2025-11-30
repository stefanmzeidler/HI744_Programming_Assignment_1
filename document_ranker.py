from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from abc import ABC, abstractmethod



class DocumentRanker(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None
        self._build()


    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def _create_vector_matrix(self):
        pass

    def top_5(self, column_name):
        """
        Inserts the top 5 most similar indices for a given matrix of vectors into a dataframe.
        :param vector_matrix: Matrix of vectors.
        :param data: Dataframe to insert top5 list.
        :param column_name: Name for new column containing the top 5.
        """

        sim_matrix = cosine_similarity(self._create_vector_matrix())
        top5_indices = np.argpartition(-sim_matrix, range(6), axis=1)[:, 1:6].tolist()
        self.dataset[column_name] = top5_indices
        self.dataset[column_name] = self.dataset[column_name].apply(lambda x: self._index_to_id(x))

    def _index_to_id(self, index_list) -> list[str]:
        """
        Given a list of indices, returns the patient UID from a given dataframe.
        :param data:Dataframe containing the patient UID.`
        :param index_list:List of indices for each patient
        :return:List of patient_ID strings
        """
        id_list = []
        for i in index_list:
            id_list.append(self.dataset.at[i, 'patient_uid'])
        return id_list
