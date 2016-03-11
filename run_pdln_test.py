#run_pdln_test.py - main file

# START HERE: http://stackoverflow.com/questions/28124366/can-gridsearchcv-be-used-with-a-custom-classifier

# SECTION : LOAD DATA
import numpy as np
from sklearn.cross_validation import train_test_split

# Fettermania libraries
import import_data
import tfidf_pdln
import plot_result_surface

# ===== SECTION: Get normalization_datadata and test/train set =====

# normalization_corpus: Series (DataFrame single column) of docs
# input_docs: DF with "cleaned_text", "original text", "doc_index"
# relevance results: DF: "cleaned_text", "original text", "doc index", "query", "relevant"
#(normalization_corpus, input_docs, relevance_results) = import_data.create_query_datasets_toy()
(normalization_corpus, input_docs, relevance_results) = import_data.create_query_datasets_crowdflower(small=True)

X_train, X_test, y_train, y_test = train_test_split(
  relevance_results[['query', 'doc_index']], relevance_results['relevant'], test_size = .3, random_state=0)

# ====== SECTION: Create model =====
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

THRESHOLD_MAX = .4
THRESHOLD_POINTS = 20
SLOPE_MAX = 1.5
SLOPE_POINTS = 30

pipe_pdln = Pipeline([('pdln', tfidf_pdln.PDLNClassifier(normalization_corpus, input_docs, verbose=True))])
param_threshold_range = np.linspace(0, THRESHOLD_MAX, THRESHOLD_POINTS)
param_slope_range = np.linspace(0, SLOPE_MAX, SLOPE_POINTS)
param_grid = [{'pdln__relevance_threshold': param_threshold_range, 'pdln__slope': param_slope_range}]

# Fettermania TODO: Is there a better scoring function? ROC?
gs = GridSearchCV(estimator=pipe_pdln, \
  param_grid=param_grid, \
  #scoring='roc_auc', \
  scoring='accuracy', \
  cv=10, \
  n_jobs=-1)

gs = gs.fit(X_train, y_train)
print('Best train score: %f' % (gs.best_score_))

# === SECTION: Evaluate training ===

results = list(
  map(
    lambda x: [
      int(round(x[0]['pdln__relevance_threshold'] * (THRESHOLD_POINTS - 1) / THRESHOLD_MAX)), \
      int(round(x[0]['pdln__slope'] * (SLOPE_POINTS - 1) / SLOPE_MAX)), \
      x[1]], gs.grid_scores_))

results_matrix = np.empty([THRESHOLD_POINTS, SLOPE_POINTS])
for (i_thr, i_slo, mean) in results:
  results_matrix[i_thr][i_slo] = mean
plot_result_surface.plot_result_surface("Accuracy", results_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
print('Best train parameters: t:%f, s:%f' % (gs.best_params_['pdln__relevance_threshold'], gs.best_params_['pdln__slope']))

# === SECTION: Evaluate best model on test ===

# Note: This doesn't give access to PDLNClassifier directly.
#pdln_classifier = gs.best_estimator_.fit(X_train, y_train)
(best_relevance_threshold, best_slope) = gs.best_params_['pdln__relevance_threshold'], gs.best_params_['pdln__slope']
pdln_classifier = tfidf_pdln.PDLNClassifier(normalization_corpus, input_docs, relevance_threshold=best_relevance_threshold, slope=best_slope)
print('Test score: %f' % (sum(pdln_classifier.predict(X_test) == y_test) / len(X_test)))


# ====== SECTION: DEBUG ======

print ("DEBUG QUERIES")
debug_queries = np.array(['cat', 'cat dog', 'felid', 'and', 'fasdf'])
print(list(zip(debug_queries, pdln_classifier.get_top_document_matches(debug_queries, 3))))
print(list(zip(debug_queries, pdln_classifier.get_documents_over_threshold(debug_queries))))




