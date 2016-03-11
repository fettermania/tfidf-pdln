#tfidf.py
import pandas as pd
import re
import numpy as np
import sklearn.feature_extraction.text as sktext
import sklearn.preprocessing as skpre
import functools

import clean_text

class TFIDFRanker(object):

  # Fettermania TODO: Why won't these stick to the instance?
  input_docs = None
  normalization_corpus = None

  # Note: Will not have pivot behavior until set_pivot() or add_normalization_corpus() called
  def __init__(self, normalization_corpus=None, input_docs=None, slope=1, pivot=1, relevance_threshold=.05):
    # print ("INIT CALLED BEGIN, id %d " % (id(self)))
    # if input_docs is None:
    #   input_docs = np.array()
    #   print ("init called with blank input docs")
    # else:
    #   print ("INPUT DOCS OK")
    # if normalization_corpus is None:
    #   normalization_corpus = pd.DataFrame()
    #   print ("init called with blank norm corpus")
    # else:
    #   print ("NORM CORPUS OK")

    # print ("INIT CALLED WITH SLOPE %d" % (slope)) 
    if normalization_corpus is not None:   
      TFIDFRanker.normalization_corpus = normalization_corpus

    if input_docs is not None:   
      TFIDFRanker.input_docs = input_docs

    print ("NORM CORPUS IS")
    print (" %d" % (TFIDFRanker.normalization_corpus.shape[0]))

    print ("INPUT DOCS IS ")
    print ("%d" % (TFIDFRanker.input_docs.shape[0]))
        
    self.slope = slope
    self.pivot = pivot
    self.relevance_threshold = relevance_threshold
    
    (self.cv, _tf_array, _df_array, _n_d, self.idf, self.tfidf_unnormalized) = self._generate_tfidf_model_unnormalized(TFIDFRanker.input_docs['cleaned_text'])
    self.tfidf = skpre.normalize(self.tfidf_unnormalized, axis=1)
    
    self.add_normalization_corpus(TFIDFRanker.normalization_corpus)
    

    print ("CALLED INIT on object %d" % (id(self)))
    print ("CALLED INIT WITH s:%f, p:%f, t:%f" % (slope, pivot, relevance_threshold))
    print ("WITH INPUT DOCS")
    if input_docs is None:
      print ("NONE")
    else: 
      print (input_docs.shape[0])
      
  
  # Note: Particular TF-IDF 
  # TF = term count
  # IDF = log(N/n_j)
  # TF-IDF = TF*IDF
  def _generate_tfidf_model_unnormalized(self, docs):
    cv = sktext.CountVectorizer(ngram_range=(1,1))
    tf_array = cv.fit_transform(docs).toarray()
    df_array = np.sum(tf_array > 0, axis=0)
    n_d = docs.size
    idf = np.log(n_d / df_array)
    tfidf_unnormalized = tf_array * idf
    return (cv, tf_array, df_array, n_d, idf, tfidf_unnormalized)

  def set_relevance_threshold(self, relevance_threshold):
    self.relevance_threshold = relevance_threshold

  def new_normalization(self, old_normalization, pivot, slope):
    return (1.0 - slope) * pivot + slope * old_normalization

  def set_pivot(self, pivot):
    self.pivot = pivot
    self._update_tfidf();
    
  def set_slope(self, slope):
    self.slope = slope
    self._update_tfidf();

  # changes in pivot or slope update tfidf
  def _update_tfidf(self):
    old_normalization = np.linalg.norm(self.tfidf_unnormalized, axis=1)
    new_normalization = self.new_normalization(old_normalization, self.pivot, self.slope)
    self.tfidf = np.empty(self.tfidf_unnormalized.shape)
    for i in range(new_normalization.size):
      self.tfidf[i] = self.tfidf_unnormalized[i] / new_normalization[i]

  def add_normalization_corpus(self, normalization_corpus):
    (cv, tf_array, df_array, n_d, idf, tfidf) = self._generate_tfidf_model_unnormalized(normalization_corpus)
    corpus_average_length = np.average(np.linalg.norm(tfidf, axis=1))
    self.set_pivot(corpus_average_length)

  def queries_to_tfidfs(self, queries, normalize=True):
    query_tf_arrays = self.cv.transform(queries).toarray();
    query_tfidfs = query_tf_arrays * self.idf
    # Fettermania TODO: In pivot world, I don't think this norm changes.
    if normalize:
      query_tfidfs = skpre.normalize(query_tfidfs, axis=1)
    return query_tfidfs

  def _get_query_scores(self, queries):
    cleaned_queries = list(map(clean_text.clean_text, queries))
    query_tfidfs = self.queries_to_tfidfs(cleaned_queries)
    vec_size = self.tfidf.shape[0]
    return np.dot(query_tfidfs, self.tfidf.transpose())

  def get_top_document_matches(self, queries, n=3):
    query_scores = self._get_query_scores(queries)
    n = min(n, query_scores.size)
    indices_top_n = np.argsort(query_scores)[:,::-1][:,:n]
    result = []
    for (array, indices) in zip(query_scores, indices_top_n):
      df = pd.DataFrame(TFIDFRanker.input_docs.iloc[indices])
      df['query_score'] = array[indices]
      result.append(df)
    return result

  def get_documents_over_threshold(self, queries):
    query_scores = self._get_query_scores(queries)
    indices_above_threshold = list(map(lambda qs: [i for i,v in enumerate(qs > self.relevance_threshold) if v], query_scores))      
    result = []
    for (array, indices) in zip(query_scores, indices_above_threshold):
      df = pd.DataFrame(TFIDFRanker.input_docs.iloc[indices])
      df['query_score'] = array[indices]
      result.append(df)
    return result

  def predict(self, queries_and_doc_indexes):
    print ("QUERIES")
    print (queries_and_doc_indexes)
    print (queries_and_doc_indexes.shape)
    print ("GETTING THAT SHAPE")
    print (queries_and_doc_indexes.iloc[:,0])
    queries = queries_and_doc_indexes.iloc[:,0]
    doc_indexes = queries_and_doc_indexes.iloc[:,1]
    doc_results = self.get_documents_over_threshold(queries)  
    final_results = np.empty(len(doc_results))
    # Fettermania: TODD - there has to be a faster way
    for i, (doc_index, doc_result) in enumerate(zip(doc_indexes, doc_results)):
      final_results[i] = (1 if doc_index in doc_result.index else 0)
    return final_results

  # Fettermania TODO this isn't well-defined as 0, 1.  Might be only for classifier
  def predict_proba(self, query_and_doc_index):
    result = self.predict(query_and_doc_index)
    return np.array([0, 1]) if result[0] == 1 else np.array([1, 0])

  def fit(self, X_train, y_train):
    # Fettermania: fit_params what are those? TODO
    print ("FIT TODO")
    return self

  def partial_fit(self, X_train, y_train):
    print ("PARTIAL FIT TODO")
    return self

  def get_params(self, deep=True):
    return {"slope" : self.slope, "relevance_threshold" : self.relevance_threshold, "pivot": self.pivot}
    #, "normalization_corpus:": TFIDFRanker.normalization_corpus, "input_docs": TFIDFRanker.input_docs}

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
      print ("set params called on object %d with" % (id(self)))
      print (parameter)
      print (value)
    return self

