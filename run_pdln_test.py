#pivot_norm.py - main file

# START HERE: http://stackoverflow.com/questions/28124366/can-gridsearchcv-be-used-with-a-custom-classifier

# SECTION : LOAD DATA
import pandas as pd
import re
import numpy as np
import sklearn.feature_extraction.text as sktext
import sklearn.preprocessing as skpre
import functools
from sklearn.cross_validation import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import datetime

# Fettermania libraries
import import_data
import tfidf_pdln
import plot_result_surface

# ===== SECTION: Get normalization_datadata and test/train set =====

# normalization_corpus: Series (array-like)
# input_docs: DF with "cleaned_text", "original text", "doc_index"
# relevance results: DF: "cleaned_text", "original text", "doc index", "query", "relevant"
#(normalization_corpus, input_docs, relevance_results) = import_data.create_query_datasets_toy()
(normalization_corpus, input_docs, relevance_results) = import_data.create_query_datasets_crowdflower(small=True)

X_train, X_test, y_train, y_test = train_test_split(
  relevance_results[['query', 'doc_index']], relevance_results['relevant'], test_size = .3, random_state=0)

# Fettermania TODO Clean when you learn pandas mo betta
X_train_array = np.array(X_train)
y_train_array = np.array(y_train)
X_test_array = np.array(X_test)
y_test_array = np.array(y_test)

# ====== SECTION: Create model =====
# # tfidf_ranker = tfidf_pdln.TFIDFRanker(input_docs=input_docs, normalization_corpus=normalization_corpus, slope=.75, relevance_threshold=.05)
# # tfidf_ranker.add_normalization_corpus(normalization_corpus)
# print ("DEBUG: Pivot is calculated at ", tfidf_ranker.pivot)

# # ===== SECTION: Run test ======
# THRESHOLD_MAX = .4
# THRESHOLD_POINTS = 20
# SLOPE_MAX = 1.5
# SLOPE_POINTS = 30

# def run_test(slope, threshold, X_array, y_array):
#   # predicted = np.empty(len(y_array))
#   tfidf_ranker.set_slope(slope)
#   tfidf_ranker.set_relevance_threshold(threshold)
#   predicted = tfidf_ranker.predict(X_array)    
#   # Fettermania TODO: Is there a different precision/recall definition for IR?
#   # https://en.wikipedia.org/wiki/Precision_and_recall
#   tp = sum(np.logical_and(predicted, y_train_array))
#   fp = sum(np.logical_and(predicted, np.logical_not(y_train_array)))
#   fn = sum(np.logical_and(np.logical_not(predicted), y_train_array))
#   accuracy  = sum(predicted == y_train_array) / len(X_array)
#   precision = 0 if tp + fp == 0 else tp / (tp + fp)
#   recall = 0 if tp + fn == 0 else tp / (tp + fn)
#   fscore = 0 if precision + recall == 0 else 2*(precision * recall)/(precision + recall)
#   print ("(DEBUG: %s) Run test: train = [a: %f, p: %f, r: %f, f: %f], slope=(%f), threshold=(%f)" % (
#     datetime.datetime.now(), accuracy, precision, recall, fscore, slope, threshold))
#   return (accuracy, precision, recall)

# accuracy_matrix = np.empty([THRESHOLD_POINTS, SLOPE_POINTS])
# precision_matrix = np.empty([THRESHOLD_POINTS, SLOPE_POINTS])
# recall_matrix = np.empty([THRESHOLD_POINTS, SLOPE_POINTS])
# # TODO: Replace with grid thing or cross-validation
# for threshold_ix in range(THRESHOLD_POINTS):
#   for slope_ix in range(SLOPE_POINTS):
#     threshold = threshold_ix * THRESHOLD_MAX / THRESHOLD_POINTS
#     slope = slope_ix * SLOPE_MAX / SLOPE_POINTS
#     (acc, prec, rec) = run_test(slope, threshold, X_train_array, y_train_array)
#     accuracy_matrix[threshold_ix][slope_ix] = acc
#     precision_matrix[threshold_ix][slope_ix] = prec
#     recall_matrix[threshold_ix][slope_ix] = rec

# # ==== SECTION: Show results ======
# plot_result_surface.plot_result_surface("Accuracy", accuracy_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
# plot_result_surface.plot_result_surface("Precision", precision_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
# plot_result_surface.plot_result_surface("Recall", recall_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

pipe_svc = Pipeline([('pdln', tfidf_pdln.TFIDFRanker(normalization_corpus, input_docs, slope=.99, pivot=.99, relevance_threshold=.99))])
param_threshold_range = np.linspace(0, .4, 20)
param_slope_range = np.linspace(0, 1.5, 30)
print ("GOT INPUT DOCS LENGTH BEFORE START: %d" % (input_docs.shape[0]))
param_grid = [{'pdln__relevance_threshold': param_threshold_range, \
#  'pdln__slope': param_slope}]
  'pdln__slope': param_slope_range, \
  #'pdln__input_docs': [input_docs], \
  'pdln__pivot': [37.2]}]
  #, \
  #'pdln__normalization_corpus': [normalization_corpus]}] 

print ("=== STARTING GRID SEARCH ===")
gs = GridSearchCV(estimator=pipe_svc, \
  param_grid = param_grid, \
  scoring ='accuracy', \
  cv = 10, \
  n_jobs = 1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_)


# ====== SECTION: DEBUG ======
print ("DEBUG QUERIES")
debug_queries = np.array(['cat', 'cat dog', 'felid', 'and', 'fasdf'])
print(list(zip(debug_queries, tfidf_ranker.get_top_document_matches(debug_queries, 3))))
print(list(zip(debug_queries, tfidf_ranker.get_documents_over_threshold(debug_queries))))




