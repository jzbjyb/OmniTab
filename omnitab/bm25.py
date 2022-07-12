from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
  # https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
  def __init__(self, b=0.75, k1=1.6):
    self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
    self.b = b
    self.k1 = k1

  def fit(self, X):
    self.vectorizer.fit(X)
    self.X = super(TfidfVectorizer, self.vectorizer).transform(X)
    self.len_X = self.X.sum(1).A1
    self.avdl = self.X.sum(1).mean()

  def transform(self, q) -> List[float]:
    b, k1, avdl = self.b, self.k1, self.avdl

    # apply CountVectorizer
    q, = super(TfidfVectorizer, self.vectorizer).transform([q])
    assert sparse.isspmatrix_csr(q)

    # convert to csc for better column slicing
    X = self.X.tocsc()[:, q.indices]
    denom = X + (k1 * (1 - b + b * self.len_X / avdl))[:, None]
    # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
    # to idf(t) = log [ n / df(t) ] with minus 1
    idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
    numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
    return (numer / denom).sum(1).A1

class BM25Wrapper(object):
  def __init__(self, docs: List[str]):
    self.docs = docs
    self.bm25 = BM25()
    self.bm25.fit(docs)

  def query(self, query: str, topk: int = 1):
    scores = self.bm25.transform(query)
    if topk == 1:
      return [self.docs[np.argmax(scores)]]
    return [self.docs[r] for r in np.argsort(-np.array(scores))[:topk]]
