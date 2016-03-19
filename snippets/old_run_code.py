# # tfidf_ranker = tfidf_pdln.TFIDFRanker(input_docs=input_docs, normalization_corpus=normalization_corpus, slope=.75, relevance_threshold=.05)
# # tfidf_ranker.add_normalization_corpus(normalization_corpus)
# print ("DEBUG: Pivot is calculated at ", tfidf_ranker.pivot)

# # === SECTION: Run test ===
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

# # === SECTION: Show results ===
# plot_result_surface.plot_result_surface("Accuracy", accuracy_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
# plot_result_surface.plot_result_surface("Precision", precision_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
# plot_result_surface.plot_result_surface("Recall", recall_matrix, THRESHOLD_MAX, THRESHOLD_POINTS, SLOPE_MAX, SLOPE_POINTS)
