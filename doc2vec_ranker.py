import document_ranker
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from gensim.test.utils import get_tmpfile
import os
import multiprocessing

class Doc2VecRanker(document_ranker.DocumentRanker):
    def _build(self):
        self.fname = "my_doc2vec_model"
        if os.path.exists(self.fname):
            print("Loading existing doc2vec model")
            self.model = Doc2Vec.load(self.fname)
        else:
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
      print("Creating doc2vec model")
      print("Building doc2vec vocab")
      cores = multiprocessing.cpu_count()
      if cores > 2:
          workers = cores - 2
      else:
          workers = 1
      model = gensim.models.doc2vec.Doc2Vec(workers=workers, min_count=1, epochs=40)
      train_corpus = list(self._read_corpus())
      model.build_vocab(train_corpus)
      print("Doc2Vec Vocab built")
      print("Doc2Vec Starting training")
      model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
      print("Doc2Vec Training Finished")
      temp_fname = get_tmpfile(os.path.join(os.getcwd(),self.fname))
      model.save(temp_fname)
      print("Model saved")
      return model

    def _create_vector_matrix(self):
      vector_matrix = []
      print("Creating Doc2Vec Vector Matrix")
      for _ , tokens in self.dataset['tokens'].items():
        vector_matrix.append(np.array(self.model.infer_vector(tokens)))
      print("Vector Matrix created")
      return vector_matrix