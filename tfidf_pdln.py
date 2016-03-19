#pdln_classifier.py
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as sktext
import sklearn.preprocessing as skpre
import datetime

# Fettermania libaries
import clean_text

class PDLNClassifier(object):

  # Fettermania TODO: new object instantiated for every gridsearchcv matrix element.
  # This is hacky but prevents complete re-generation of the TFs and DFs.
  # This prevents, however, more than one PDLNClassifier from operating with different
  # corpuses
  input_docs = None
  tfidf_unnormalized = None
  idf = None
  cv = None
  pivot = None
  verbose = False

  def __init__(self, normalization_corpus=None, input_docs=None, slope=1, relevance_threshold=.05, verbose=False):
    # Fettermania note: There's some weird instantiation behavior in GridSearchCV 
    # requiring this strange static variable kung fu.  TODO fix.
    if input_docs is not None and normalization_corpus is not None:
      PDLNClassifier.input_docs = input_docs
      (PDLNClassifier.cv, _tf_array, _df_array, _n_d, PDLNClassifier.idf, PDLNClassifier.tfidf_unnormalized) = self._generate_tfidf_model_unnormalized(PDLNClassifier.input_docs['cleaned_text'])
      PDLNClassifier.pivot = self.pivot_from_normalization_corpus(normalization_corpus)
    if verbose:
      PDLNClassifier.verbose = True

    self.slope = slope
    self.relevance_threshold = relevance_threshold
    self.pivot = PDLNClassifier.pivot
    self._update_tfidf()

  # === SECTION: Math ==

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

  def _new_normalization(self, old_normalization, pivot, slope):
    return (1.0 - slope) * pivot + slope * old_normalization

  # changes in pivot or slope update tfidf
  def _update_tfidf(self):
    old_normalization = np.linalg.norm(PDLNClassifier.tfidf_unnormalized, axis=1)
    new_normalization = self._new_normalization(old_normalization, self.pivot, self.slope)
    self.tfidf = np.empty(PDLNClassifier.tfidf_unnormalized.shape)
    for i in range(new_normalization.size):
      self.tfidf[i] = PDLNClassifier.tfidf_unnormalized[i] / new_normalization[i]

  def pivot_from_normalization_corpus(self, normalization_corpus):
    (cv, tf_array, df_array, n_d, idf, tfidf) = self._generate_tfidf_model_unnormalized(normalization_corpus)
    corpus_average_length = np.average(np.linalg.norm(tfidf, axis=1))
    return corpus_average_length

  # === SECTION: Getters/setters ===

  def set_pivot(self, pivot):
    self.pivot = pivot
    self._update_tfidf();
    
  def set_slope(self, slope):
    self.slope = slope
    self._update_tfidf();

  def set_relevance_threshold(self, relevance_threshold):
    self.relevance_threshold = relevance_threshold

  def get_params(self, deep=True):
    return {"slope" : self.slope, "relevance_threshold" : self.relevance_threshold}

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      if parameter == 'slope':
        self.set_slope(value)
      elif parameter == 'pivot':
        self.set_pivot(value)
      else:
        setattr(self, parameter, value)
    return self

  # === SECTION: Learning (no-op) ===
  # No-op.  No learning here.
  def fit(self, X_train, y_train):
    return self

  # No-op.  No learning here.
  def partial_fit(self, X_train, y_train):
    # print ("PARTIAL FIT TODO")
    return self


  # === SECTION: Prediction ===
  def _queries_to_tfidfs(self, queries, normalize=True):
    query_tf_arrays = self.cv.transform(queries).toarray();
    query_tfidfs = query_tf_arrays * PDLNClassifier.idf
    # Fettermania TODO: In pivot world, I don't think this norm changes.
    if normalize:
      query_tfidfs = skpre.normalize(query_tfidfs, axis=1)
    return query_tfidfs

  def _get_query_scores(self, queries):
    cleaned_queries = list(map(clean_text.clean_text, queries))
    query_tfidfs = self._queries_to_tfidfs(cleaned_queries)
    vec_size = self.tfidf.shape[0]
    return np.dot(query_tfidfs, self.tfidf.transpose())

  def get_top_document_matches(self, queries, n=3):
    query_scores = self._get_query_scores(queries)
    n = min(n, query_scores.size)
    indices_top_n = np.argsort(query_scores)[:,::-1][:,:n]
    result = []
    for (array, indices) in zip(query_scores, indices_top_n):
      df = pd.DataFrame(PDLNClassifier.input_docs.iloc[indices])
      df['query_score'] = array[indices]
      result.append(df)
    return result

  def get_documents_over_threshold(self, queries):
    query_scores = self._get_query_scores(queries)
    indices_above_threshold = list(map(lambda qs: [i for i,v in enumerate(qs > self.relevance_threshold) if v], query_scores))      
    result = []
    for (array, indices) in zip(query_scores, indices_above_threshold):
      df = pd.DataFrame(PDLNClassifier.input_docs.iloc[indices])
      df['query_score'] = array[indices]
      result.append(df)
    return result

  def predict(self, queries_and_doc_indexes):
    if PDLNClassifier.verbose:
      print("%s: Run predict on (thr:%f, slo:%f)" % (datetime.datetime.now(), self.relevance_threshold, self.slope))

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

