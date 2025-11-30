from sklearn.feature_extraction.text import TfidfVectorizer
import document_ranker

class TFIDFRanker(document_ranker.DocumentRanker):
    def _build(self):
        print("Creating TFIDF ranker...")
        self.model = TfidfVectorizer()
        print("TFIDF ranker created")
    def _create_vector_matrix(self):
        print("Creating TFIDF vector matrix...")
        vector_matrix = self.model.fit_transform(self.dataset['patient'])
        print("TFIDF vector matrix created")
        return vector_matrix
