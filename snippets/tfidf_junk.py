


# def tfidf_calculate(input_docs, tfidf_f, normalize=True):
#   if tfidf_f == tfidf_builtin:
#     return tfidf_builtin(input_docs)
#   tfidf = tfidf_f(input_docs)
#   if normalize:
#     tfidf = skpre.normalize(tfidf, axis=1)
#   return tfidf

# # # Variant of tfidf
# # def tfidf_1(input_docs):
# #   idf = np.log(n_d / (1 + df_array))
# #   tfidf = tf_array * (idf + 1)
# #   return tfidf

# # # Variant of tfidf, equivalent to scikit builtin
# # # Note: Whole tf-idf normalized here, not just tf
# # def tfidf_2(input_docs):
# #   idf_2 = np.log((1 + n_d)/ (1 + df_array))
# #   tfidf_2 = tf_array * (idf_2 + 1)
# #   return tfidf_2

# # # TFIDF builtin, for confirmation
# # def tfidf_builtin(input_docs):
# #   cv_fitted = cv.fit_transform(input_docs)
# #   tfidf_builtin_transformer = sktext.TfidfTransformer()
# #   tfidf_builtin = tfidf_builtin_transformer.fit_transform(cv_fitted)
# #   tfidf_builtin_array = tfidf_builtin.toarray()
# #   return tfidf_builtin_array
