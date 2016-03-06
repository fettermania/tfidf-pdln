from nltk.stem.porter import *
 
stemmer = PorterStemmer()

def clean_query_string(query):
  query = query.strip()
  query = re.sub("[\W]+", " ", query)
  return porter_stem_pass(query)

def clean_text(text):
  text = text.strip()
  text = re.sub('<[^>]*>', '', text)
  text = re.sub('\'', '', text)
  text = re.sub('[\W]+', ' ', text)
  text = text.lower()
  return porter_stem_pass(text)

def porter_stem_pass(text):
  return ' '.join([stemmer.stem(i) for i in text.split(' ')])