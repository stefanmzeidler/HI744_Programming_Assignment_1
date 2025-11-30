import document_ranker
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

class Doc2VecRanker(document_ranker.DocumentRanker):
    def _build(self):
        self.model = self._train_doc2vec()

    def _read_corpus(self, tokens_only=False):
        """
        Adapted from https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py.
        Reads in tokens column from dataset dataframe as the corpus for training the gensim model.
        :param tokens_only: Whether to yield only the tokens or include an index label.
        """
        documents = self.dataset['tokens']
        for i, tokens in documents.items():
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def _train_doc2vec(self):
      model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=40)
      train_corpus = list(self._read_corpus())
      model.build_vocab(train_corpus)
      print("Doc2Vec Vocab built")
      print("Doc2Vec Starting training")
      model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
      print("Doc2Vec Training Finished")
      return model

    def _create_vector_matrix(self):
      vector_matrix = []
      print("Creating Doc2Vec Vector Matrix")
      for _ , tokens in self.dataset['tokens'].items():
        vector_matrix.append(np.array(self.model.infer_vector(tokens)))
      print("Vector Matrix created")
      return vector_matrix