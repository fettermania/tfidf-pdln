#run_pdln_test.py - main file

#2016-03-11 23:27:02.364828: Run predict on (thr:0.147368, slo:0.827586)
#Test score: 0.637719

import sys, getopt
from sklearn.externals import joblib
import sklearn.metrics
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

# Fettermania libraries
import import_data
import pdln_classifier
import plot_results


# Fettermania: TODO move "globals" to config
THRESHOLD_MAX = .4
THRESHOLD_POINTS = 20
SLOPE_MAX = 1.5
SLOPE_POINTS = 30

# === STRATEGY 1: FULL grid search of threshold, slope ===
def run_full_2d_search(input_args, data_args, cv, result_metric, n_jobs, verbose):
  (normalization_corpus, input_docs, relevance_results) = input_args
  (X_train, X_test, y_train, y_test) = data_args

  # === Create pipeline ===
  pipe_pdln = Pipeline([('pdln', pdln_classifier.PDLNClassifier(normalization_corpus, input_docs, verbose=verbose))])
  param_threshold_range = np.linspace(0, THRESHOLD_MAX, THRESHOLD_POINTS)
  param_slope_range = np.linspace(0, SLOPE_MAX, SLOPE_POINTS)
  param_grid = [{'pdln__relevance_threshold': param_threshold_range, 'pdln__slope': param_slope_range}]

  gs = GridSearchCV(estimator=pipe_pdln, \
    param_grid=param_grid, \
    scoring=result_metric, \
    cv=cv, \
    n_jobs=n_jobs)

  # === Search for best hyperparameter fit ===
  gs = gs.fit(X_train, y_train)

  # === Evaluate training ===
  print('Best train score (%s): %f at slope = %f, threshold: %f' % (result_metric, gs.best_score_, gs.best_params_['pdln__slope'], gs.best_params_['pdln__relevance_threshold']))

  results = list(
    map(
      lambda x: [
        int(round(x[0]['pdln__relevance_threshold'] * (THRESHOLD_POINTS - 1) / THRESHOLD_MAX)), \
        int(round(x[0]['pdln__slope'] * (SLOPE_POINTS - 1) / SLOPE_MAX)), \
        x[1]], gs.grid_scores_))

  results_matrix = np.empty([THRESHOLD_POINTS, SLOPE_POINTS])
  for (i_thr, i_slo, mean) in results:
    results_matrix[i_thr][i_slo] = mean
  plot_results.plot_result_surface(result_metric, results_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
  (best_relevance_threshold, best_slope) = gs.best_params_['pdln__relevance_threshold'], gs.best_params_['pdln__slope']
  return (gs.best_score_, best_relevance_threshold, best_slope)

# === STRATEGY 2: Find optimal threshold, then slope ===
def run_two_step_search(input_args, data_args, cv, result_metric, n_jobs, verbose):
  (normalization_corpus, input_docs, relevance_results) = input_args
  (X_train, X_test, y_train, y_test) = data_args

  # === Search for best threshold fit given fixed slope=1 ===
  pipe_pdln = Pipeline([('pdln', pdln_classifier.PDLNClassifier(normalization_corpus, input_docs, verbose=verbose))])
  param_threshold_range = np.linspace(0, THRESHOLD_MAX, THRESHOLD_POINTS)
  param_grid = [{'pdln__relevance_threshold': param_threshold_range, 'pdln__slope': [1.0]}]

  print('Optimizing threshold at slope = 1...')
  gs = GridSearchCV(estimator=pipe_pdln, \
    param_grid=param_grid, \
    scoring=result_metric, \
    cv=cv, \
    n_jobs=n_jobs)

  gs = gs.fit(X_train, y_train)
  print('Best train score (%s): %f at slope = %f, threshold: %f' % (result_metric, gs.best_score_, gs.best_params_['pdln__slope'], gs.best_params_['pdln__relevance_threshold']))

  threshold_scores_x = list(map(lambda x: x[0]['pdln__relevance_threshold'], gs.grid_scores_))
  threshold_scores_y = list(map(lambda x: x[1], gs.grid_scores_))

  # === Search for best slope fit given fixed optimal threshold ===
  param_slope_range = np.linspace(0, SLOPE_MAX, SLOPE_POINTS)
  param_grid = [{'pdln__relevance_threshold': [gs.best_params_['pdln__relevance_threshold']], 'pdln__slope': param_slope_range}]

  print('Optimizing slope at relevance = %f...' % (gs.best_params_['pdln__relevance_threshold']))
  gs = GridSearchCV(estimator=pipe_pdln, \
    param_grid=param_grid, \
    scoring=result_metric, \
    cv=cv, \
    n_jobs=n_jobs)
  gs = gs.fit(X_train, y_train)

  slope_scores_x = list(map(lambda x: x[0]['pdln__slope'], gs.grid_scores_))
  slope_scores_y = list(map(lambda x: x[1], gs.grid_scores_))


  plot_results.plot_1d_search_results(result_metric, gs.best_params_['pdln__relevance_threshold'], threshold_scores_x, threshold_scores_y, slope_scores_x, slope_scores_y)
  (best_relevance_threshold, best_slope) = gs.best_params_['pdln__relevance_threshold'], gs.best_params_['pdln__slope']
  return (gs.best_score_, best_relevance_threshold, best_slope)



def run_pdln_test(cv, dataset, result_metric, n_jobs, full2d, verbose):

  # === Get normalization_datadata and test/train set ===

  # normalization_corpus: Series (DataFrame single column) of docs
  # input_docs: DF with "cleaned_text", "original text", "doc_index"
  # relevance results: DF: "cleaned_text", "original text", "doc index", "query", "relevant"
  if dataset == "small":
    input_args = import_data.create_query_datasets_crowdflower(small=True)
  elif dataset == "large":
    input_args = import_data.create_query_datasets_crowdflower(small=False)
  else: 
    input_args = import_data.create_query_datasets_toy()


  (normalization_corpus, input_docs, relevance_results) = input_args
  data_args = train_test_split(
    relevance_results[['query', 'doc_index']], relevance_results['relevant'], test_size = .3, random_state=0)
  (X_train, X_test, y_train, y_test) = data_args
  

  # Fettermania: Memory map data files
  os.system("rm -f X_train.tmp* y_train.tmp*")
  joblib.dump(X_train, "./X_train.tmp")
  X_train = joblib.load("./X_train.tmp", mmap_mode='r+')
  joblib.dump(y_train, "./y_train.tmp")
  y_train = joblib.load("./y_train.tmp", mmap_mode='r+')

  try:
    if full2d:
      (best_score, best_relevance_threshold, best_slope) = run_full_2d_search(input_args, data_args, cv, result_metric, n_jobs, verbose)
    else:
      (best_score, best_relevance_threshold, best_slope) = run_two_step_search(input_args, data_args, cv, result_metric, n_jobs, verbose)
  finally:
    os.system("rm -f X_train.tmp* y_train.tmp*")



  # Note: This doesn't give access to PDLNClassifier directly.
  #pdln_classifier = gs.best_estimator_.fit(X_train, y_train)
  classifier = pdln_classifier.PDLNClassifier(normalization_corpus, input_docs, relevance_threshold=best_relevance_threshold, slope=best_slope)


  from sklearn.metrics import confusion_matrix
  y_pred = classifier.predict(X_train)
  confmat = confusion_matrix(y_true=y_train, y_pred=y_pred)
  print("Training confusion matrix:\n")
  print(confmat)

  # === Evaluate best model on test ===
  test_score =  getattr(sklearn.metrics, result_metric + "_score")(
    classifier.predict(X_test),
    y_test)


  y_pred = classifier.predict(X_test)
  confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
  print("Test confusion matrix:\n")
  print(confmat)

  print('Test score (%s) at threshold=%f, slope=%f: %f' % (result_metric, best_relevance_threshold, best_slope, test_score))





  # === DEBUG/play section ===
  # print ("DEBUG QUERIES")
  # debug_queries = np.array(['cat', 'cat dog', 'felid', 'and', 'fasdf'])
  # print(list(zip(debug_queries, classifier.get_top_document_matches(debug_queries, 3))))
  # print(list(zip(debug_queries, classifier.get_documents_over_threshold(debug_queries))))



# === MAIN program ===
def usage():
  print('python run_pdln_test.py [-c <folds>] [-d <toy|small|large>] [-m <result_metric>] [-n n_jobs] [--full2d] [--verbose]')


def main(argv):
  cv = 3
  dataset = "small"
  result_metric = "accuracy"
  n_jobs = 1
  full2d = False
  verbose = False

  try:
    opts, args = getopt.getopt(argv,"c:d:m:n:",["full2d", "verbose"])
  except getopt.GetoptError:
    usage()
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-c':
      cv = int(arg)
      if cv < 2:
        print("error: provided argument c < 2")
        usage()
        sys.exit(2)
    elif opt == '-d':
      if arg not in ("toy", "small", "large"):
        print('error: provided argument d not in set ("toy", "small", "large")')
        usage()
        sys.exit(2)
      dataset = arg
    elif opt == '-m':
      # List at http://scikit-learn.org/stable/modules/model_evaluation.html
      if not hasattr(sklearn.metrics, arg + "_score"):
        print ("error: result metric needs have method <result_metric>_score defined in sklearn.metrics")
        usage()
        sys.exit(2)
      result_metric = arg
    elif opt == '-n':
      n_jobs = int(arg)
      if n_jobs < 1 and n_jobs != -1:
        print("error: provided argument n < 1 and not -1")
        usage()
        sys.exit(2)
    elif opt == '--full2d':
      full2d = True
    elif opt == '--verbose':
      verbose = True
  print ("Params: cv=%d, dataset=%s, result_metric=%s, n_jobs=%d, full2d=%r, verbose=%r" % (
      cv, dataset, result_metric, n_jobs, full2d, verbose))

  run_pdln_test(cv, dataset, result_metric, n_jobs, full2d, verbose)

if __name__ == "__main__":
   main(sys.argv[1:])




