Pivoted Document Length Normalization
===

To Use:
---

- choose dataset, among:
  - full crowdflower data (use import_data.create_query_datasets_crowdflower(small=False), run_pdln_test:25)
    - Full evaluation can take a few hours.
  - partial crowdflower data (use import_data.create_query_datasets_crowdflower(small=False), run_pdln_test:25)
    - Quick evaluation
  - toy example data (use import_data.create_query_datasets_toy(), run_pdln_test:24)
    - Quick evaluation
- python run_pdln_test.py
- Can also play with queries in SECTION: DEBUG in run_pdln_test

Results
---
- results/crowdflower_results_large.csv: 
  - Low accuracy, very low recall suggests our thresholds might be too high.
  - TODO: Actually look at test data.
  - TODO: Actually confine to smaller region
  

Considerations
---
- Crowdflower data: Right now, we are looking at accuracy of "relevance" (crowdflower "relevance score" > 4 means "relevant retrieval", else "irrelevant retrieval").  Probably better to learn this function (range: 0-4) than collapse into a binary classificatin.

- More things to consider in docs/notes.txt

TODO
---


TODO 
---
- Remove static class variables in classifier class
- Try RandomizedSearchCV instead of GridSearchCV
- See if we can get n_jobs=2 or n_jobs=-1 to work on large dataset 
- SBM section 3: idf is used in query terms, not in doc weigths
- To optimize:
  - Use PAIRWISE rankings?
    - Note: This exponentially increases the number of "pivots" because there's no P(ret) and P(rel) curves that cross anymore.
  - Metrics (later): Old: precision, recall, f-score.  New: ROC, DCG and variants.
  - Chapelle: Judgement metric for the contest was NDCG 
- TODO: Dump to .csv
 


NOtes
---
- Memory map:  # http://stackoverflow.com/questions/24406937/scikit-learn-joblib-bug-multiprocessing-pool-self-value-out-of-range-for-i-fo/24411581#24411581
- BM25 - whole system; https://en.wikipedia.org/wiki/Okapi_BM25
  - need to learn more core constants





