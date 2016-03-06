import numpy as np
import sklearn.feature_extraction.text as sktext
import sklearn.preprocessing as skpre

# Fettermania: Can switch grams
#cv = textlib.CountVectorizer(ngram_range=(1,2))
cv = sktext.CountVectorizer(ngram_range=(1,1))

# Note: Doesn't convert very small words or punctuation
doc_freqs = np.array( ['The sun is shining', 'The weather is sweet', 'The sun is shining and the weather is sweet'])

tf = cv.fit_transform(doc_freqs);
tf_array = tf.toarray();

print ("VOCAB")
print (cv.vocabulary_)

print ("tf")
print (tf_array)

# n_d row vectors of frequencies in that doc.
# columns: size of cv.vocabulary_
df_array = np.sum(tf_array > 0, axis=0)
n_d = doc_freqs.size
#idf_1 = np.log(n_d / (1 + df_array))
#tfidf_1 = tf_array * (idf_1 + 1)
idf_2 = np.log((1 + n_d)/ (1 + df_array))
tfidf_2 = tf_array * (idf_2 + 1)
tfidf_2_normalized = skpre.normalize(tfidf_2, axis=1)


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_builtin_transformer = TfidfTransformer()
tfidf_builtin = tfidf_builtin_transformer.fit_transform(cv.fit_transform(doc_freqs))
tfidf_builtin_array = tfidf_builtin.toarray()

print ("TFIDF HANDROLL should be 0")
print (tfidf_2_normalized)
print ("TFIFD BUILTIN should be 0")
print (tfidf_builtin_array)

print ("DEBUG: DIFF should be 0")
print (tfidf_builtin_array - tfidf_2_normalized)


