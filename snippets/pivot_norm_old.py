#pivot_norm.py

# SECTION : LOAD DATA
import pandas as pd
import re
import numpy as np
import sklearn.feature_extraction.text as sktext
import sklearn.preprocessing as skpre
import functools
from sklearn.cross_validation import train_test_split

# Fettermania libraries
import import_data

# For now, just import one corpus and clean

# ===== SECTION: Get normalization_datadata and test/train set =====
(normalization_corpus, dataset, relevance_results) = import_data.create_query_datasets()

X_train, X_test, y_train, y_test = train_test_split(
  relevance_results['cleaned_text'], relevance_results['relevant'], test_size = .3, random_state=0)


# Fettermania TODO: Gotta learn how to use pandas for real
input_docs = dataset['cleaned_text'].values;
original_input_docs = dataset['original_text'].values

# TF = count
# IDF = log(N/n_j), from lec3
# Returns cv_idf_tfidf
def cv_idf_tfidf_sbm_l2_corpus(input_docs, normalize=True):
  cv = sktext.CountVectorizer(ngram_range=(1,1))
  tf_array = cv.fit_transform(input_docs).toarray()
  df_array = np.sum(tf_array > 0, axis=0)
  n_d = input_docs.size
  idf = np.log(n_d / df_array)
  tfidf = tf_array * idf
  if normalize: 
    tfidf = skpre.normalize(tfidf, axis=1)
  return (cv, idf, tfidf)

# Fettermania: May not need to normalize this.
def tfidf_vec_sbm_l2_onetime(query, cv, idf, normalize=True):
  tf_array = cv.transform([query]).toarray();
  tfidf = tf_array * idf
  # Fettermania TODO: In pivot world, I don't think this norm changes.
  if normalize:
    tfidf = skpre.normalize(tfidf, axis=1)
  return tfidf

# Fettermania TODO: We're running through the whole corpus a lot here.
def calculate_corpus_l2_normalization(input_docs):
  (cv, idf, tfidf) = cv_idf_tfidf_sbm_l2_corpus(input_docs, normalize=False)
  old_normalization = np.average(np.linalg.norm(tfidf, axis=1))
  return old_normalization

def new_normalization(old_normalization, pivot, slope):
  return (1.0 - slope) * pivot + slope * old_normalization

def cv_idf_tfidf_sbm_pivot_corpus(input_docs, pivot, slope):
  (cv, idf, tfidf) = tfidf_sbm_l2_corpus(input_docs)
  old_norm = np.linalg.norm(tfidf, axis=1)
  new_norm = new_normalization(old_norm, pivot, slope)
  for i in range(new_norm.size):
    tfidf[i] /= new_norm[i]
  return (cv, idf, tfidf)


# SECTION: Let's get to it

# Fettermania: TODO - don't cheat by using whole set 
# to generate pivot
pivot = calculate_corpus_l2_normalization(normalization_corpus['cleaned_text'])
slope = 1 # Fettermania: Default, target is *lower*
print ("DEBUG: Pivot is calculated at ", pivot)

# tfidf_f = functools.partial(
#   tfidf_sbm_pivot, tfidf_2, pivot, slope)

(cv_model, idf_model, tfidf_model) = cv_idf_tfidf_sbm_l2_corpus(input_docs, normalize=True)


# Can have duplicates, gives higher weight
# NORMALIZED, by the way
def clean_query_string(query):
  query = query.strip()
  query = re.sub("[\W]+", " ", query)
  return query

def get_query_scores(query_vec):
  vec_size = tfidf_model.shape[0]
  result = np.empty(vec_size,)
  for i in range(vec_size):
    result[i] = np.dot(query_vec, tfidf_model[i])
  return result

def get_top_n_results(query_scores, n=3):
  top_n = np.argsort(query_scores)[-n:][::-1]
  scores = query_scores[top_n]
  result = []
  for i in range(n):
    df_index = top_n[i]
    result.append((df_index, scores[i], original_input_docs[df_index]))
  return result

def run_query(query, n=3):
  cleaned_query = clean_query_string(query)
  query_tfidf = tfidf_vec_sbm_l2_onetime(cleaned_query, cv_model, idf_model, normalize=True)
  query_scores = get_query_scores(query_tfidf)
  n = min(n, query_scores.size)
  top_results = get_top_n_results(query_scores, n)
  return top_results


debug_queries = np.array(['cat', 'cat dog', 'digitigrades', 'fooof', 'cat cat dog'])

for i in range(debug_queries.size):
  print ("QUERY :", debug_queries[i])
  print(run_query(debug_queries[i], 10))

