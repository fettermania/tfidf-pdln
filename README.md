Pivoted Document Length Normalization
===

To Use:
---
- python run_pdln_test.py [-c <folds>] [-d <toy|small|large>] [-m <result_metric>] [-n n_jobs] [--full2d] [--verbose]
  - c: (cross-validation param) folds.  Default 3.
  - d: (dataset).  Default "small".  Choose among:
    - "large": full crowdflower data
      - Note: Full evaluation can take a few hours.
    - "small": partial crowdflower data
      - Quick evaluation
    - "toy": toy example data 
  - m: result metric.  Default "accuracy".  List at http://scikit-learn.org/stable/modules/model_evaluation.html
  - n: number of concurrent jobs allowed.  Default 1.
    - Note: For large datasets, anything other than 1 seems to segfault.
  - full2d option: do full search of 2d param space (threshold, slope).
  - verbose
- You can also play with queries in SECTION: DEBUG in run_pdln_test.py

Results
---
- results/crowdflower_results_large.png
  - Note: currently segfaults on full 2d search
  - On the most accurate (65%) parameters in training, the model still has very low precision (64%), though high recall (90%).  This is clear since it predicts something like 82% of results to be positive, the main issue.  
  - Dummy "guess clss 1" strategy gives 61% accuracy. TFIDF improves to 64%.
  - Slope doesn't seem to have a major effect, though it does add most accuracy in the range suggested by SBM (around .8)
- results/crowdflower_results_small.png
  - Suggests the nested 1d strategy is sensible.
  
Considerations
---
- Crowdflower data: Right now, we are looking at accuracy of "relevance" (crowdflower "relevance score" == 4 means "relevant retrieval", else "irrelevant retrieval").  Probably better to learn this function (range: 0-4) than collapse into a binary classification.  This probably causes a lot of the problems in the model.
- Also, the notion that one threshold determines "relevance" might be misapplied.


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
 


Notes
---
- Memory map:  # http://stackoverflow.com/questions/24406937/scikit-learn-joblib-bug-multiprocessing-pool-self-value-out-of-range-for-i-fo/24411581#24411581
- BM25 - whole system; https://en.wikipedia.org/wiki/Okapi_BM25
  - need to learn more core constants





