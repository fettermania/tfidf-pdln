#run_pdln_test.py - main file

#2016-03-11 23:27:02.364828: Run predict on (thr:0.147368, slo:0.827586)
#Test score: 0.637719

# SECTION : LOAD DATA
import numpy as np
from sklearn.cross_validation import train_test_split

# Fettermania libraries
import import_data
import tfidf_pdln
import plot_results

STRATEGY_RUN_FULL_2D = False
CV = 3
N_JOBS = 1 # Fettermaina TOOD: Why does N_JOBS = 2 hang?  


# ===== SECTION: Get normalization_datadata and test/train set =====

# normalization_corpus: Series (DataFrame single column) of docs
# input_docs: DF with "cleaned_text", "original text", "doc_index"
# relevance results: DF: "cleaned_text", "original text", "doc index", "query", "relevant"
#(normalization_corpus, input_docs, relevance_results) = import_data.create_query_datasets_toy()
(normalization_corpus, input_docs, relevance_results) = import_data.create_query_datasets_crowdflower(small=False)

X_train, X_test, y_train, y_test = train_test_split(
  relevance_results[['query', 'doc_index']], relevance_results['relevant'], test_size = .3, random_state=0)


# ====== SECTION: Create model =====
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

THRESHOLD_MAX = .4
THRESHOLD_POINTS = 20
SLOPE_MAX = 1.5
SLOPE_POINTS = 30

# ============ STRATEGY 1: FULL grid search of threshold, slope ===============
def run_full_2d_search():
  pipe_pdln = Pipeline([('pdln', tfidf_pdln.PDLNClassifier(normalization_corpus, input_docs, verbose=True))])
  param_threshold_range = np.linspace(0, THRESHOLD_MAX, THRESHOLD_POINTS)
  param_slope_range = np.linspace(0, SLOPE_MAX, SLOPE_POINTS)
  param_grid = [{'pdln__relevance_threshold': param_threshold_range, 'pdln__slope': param_slope_range}]

  # Fettermania TODO: Is there a better scoring function? ROC?
  gs = GridSearchCV(estimator=pipe_pdln, \
    param_grid=param_grid, \
    #scoring='roc_auc', \
    scoring='accuracy', \
    cv=CV, \
    n_jobs=N_JOBS)

  gs = gs.fit(X_train, y_train)

  # === SECTION: Evaluate training ===

  print('Best train score: %f at slope = %f, threshold: %f' % (gs.best_score_, gs.best_params_['pdln__slope'], gs.best_params_['pdln__relevance_threshold']))

  results = list(
    map(
      lambda x: [
        int(round(x[0]['pdln__relevance_threshold'] * (THRESHOLD_POINTS - 1) / THRESHOLD_MAX)), \
        int(round(x[0]['pdln__slope'] * (SLOPE_POINTS - 1) / SLOPE_MAX)), \
        x[1]], gs.grid_scores_))

  results_matrix = np.empty([THRESHOLD_POINTS, SLOPE_POINTS])
  for (i_thr, i_slo, mean) in results:
    results_matrix[i_thr][i_slo] = mean
  plot_results.plot_result_surface("Accuracy", results_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
  (best_relevance_threshold, best_slope) = gs.best_params_['pdln__relevance_threshold'], gs.best_params_['pdln__slope']
  return (best_relevance_threshold, best_slope)

# ============ STRATEGY 2: Find optimal threshold, then slope ===============
def run_two_step_search():
  # === FIRST, find ideal Threshold, slope = 1 ===
  pipe_pdln = Pipeline([('pdln', tfidf_pdln.PDLNClassifier(normalization_corpus, input_docs, verbose=True))])
  param_threshold_range = np.linspace(0, THRESHOLD_MAX, THRESHOLD_POINTS)
  param_grid = [{'pdln__relevance_threshold': param_threshold_range, 'pdln__slope': [1.0]}]

  print('Optimizing threshold at slope = 1...')
  gs = GridSearchCV(estimator=pipe_pdln, \
    param_grid=param_grid, \
    #scoring='roc_auc', \
    scoring='accuracy', \
    cv=CV, \
    n_jobs=N_JOBS)

  gs = gs.fit(X_train, y_train)
  print('Best train score: %f at slope = %f, threshold: %f' % (gs.best_score_, gs.best_params_['pdln__slope'], gs.best_params_['pdln__relevance_threshold']))

  optimal_threshold_x = list(map(lambda x: x[0]['pdln__relevance_threshold'], gs.grid_scores_))
  optimal_threshold_y = list(map(lambda x: x[1], gs.grid_scores_))

  # === SECOND, find best slope ===
  param_slope_range = np.linspace(0, SLOPE_MAX, SLOPE_POINTS)
  param_grid = [{'pdln__relevance_threshold': [gs.best_params_['pdln__relevance_threshold']], 'pdln__slope': param_slope_range}]

  print('Optimizing slope at relevance = %f...' % (gs.best_params_['pdln__relevance_threshold']))
  gs = GridSearchCV(estimator=pipe_pdln, \
    param_grid=param_grid, \
    #scoring='roc_auc', \
    scoring='accuracy', \
    cv=CV, \
    n_jobs=N_JOBS)
  gs = gs.fit(X_train, y_train)

  optimal_slope_x = list(map(lambda x: x[0]['pdln__slope'], gs.grid_scores_))
  optimal_slope_y = list(map(lambda x: x[1], gs.grid_scores_))

  print('Best train score: %f at slope = %f, threshold: %f' % (gs.best_score_, gs.best_params_['pdln__slope'], gs.best_params_['pdln__relevance_threshold']))

  plot_results.plot_1d_search_results(gs.best_params_['pdln__relevance_threshold'], optimal_threshold_x, optimal_threshold_y, optimal_slope_x, optimal_slope_y)
  (best_relevance_threshold, best_slope) = gs.best_params_['pdln__relevance_threshold'], gs.best_params_['pdln__slope']
  return (best_relevance_threshold, best_slope)

# === SECTION: Start model ===

# Fettermania: Memory map
# http://stackoverflow.com/questions/24406937/scikit-learn-joblib-bug-multiprocessing-pool-self-value-out-of-range-for-i-fo/24411581#24411581
from sklearn.externals import joblib
import os

os.system("rm -f X_train.tmp* y_train.tmp*")

joblib.dump(X_train, "./X_train.tmp")
X_train = joblib.load("./X_train.tmp", mmap_mode='r+')

joblib.dump(y_train, "./y_train.tmp")
y_train = joblib.load("./y_train.tmp", mmap_mode='r+')

if STRATEGY_RUN_FULL_2D:
  (best_relevance_threshold, best_slope) = run_full_2d_search()
else:
  (best_relevance_threshold, best_slope) = run_two_step_search()

os.system("rm -f X_train.tmp* y_train.tmp*")



# === SECTION: Evaluate best model on test ===

# Note: This doesn't give access to PDLNClassifier directly.
#pdln_classifier = gs.best_estimator_.fit(X_train, y_train)
pdln_classifier = tfidf_pdln.PDLNClassifier(normalization_corpus, input_docs, relevance_threshold=best_relevance_threshold, slope=best_slope)
print('Test score: %f' % (sum(pdln_classifier.predict(X_test) == y_test) / len(X_test)))

# ====== SECTION: DEBUG ======

print ("DEBUG QUERIES")
debug_queries = np.array(['cat', 'cat dog', 'felid', 'and', 'fasdf'])
print(list(zip(debug_queries, pdln_classifier.get_top_document_matches(debug_queries, 3))))
print(list(zip(debug_queries, pdln_classifier.get_documents_over_threshold(debug_queries))))




