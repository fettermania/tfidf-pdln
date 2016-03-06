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




