# Clinical NLP Tutorial with MIMIC-III

A reproducible clinical NLP tutorial that compares spaCy, scispaCy, and medSpaCy on a disease-focused MIMIC-III discharge-summary cohort. The workflow extracts clinical entities, trains Word2Vec embeddings, tunes t-SNE with trustworthiness, and saves aggregate results only.

## What this tutorial teaches

- How to create a MIMIC-III cohort from `DIAGNOSES_ICD` and `NOTEEVENTS`.
- How spaCy, scispaCy, and medSpaCy differ in a controlled entity-extraction workflow.
- How to train Word2Vec on per-note entity sequences.
- How to tune t-SNE with a fixed seed and trustworthiness score.
- How to interpret exploratory visualizations responsibly.

## Data access and privacy

This repository does **not** include MIMIC-III data, raw note text, patient-level outputs, or credentials. Running the tutorial requires authorized access to the full MIMIC-III database and compliance with its data-use agreement. Do not publish any MIMIC note text or patient-level data.

## Setup

1. Create a Python environment (Python 3.10+ recommended).
2. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain authorized access to the full MIMIC-III database.
4. In `Self_Learning_Clinical_NLP_Tutorial.py`, set `MIMIC_ROOT` to the folder containing the authorized `DIAGNOSES_ICD.csv.gz` and `NOTEEVENTS.csv.gz` files.
5. Run:

   ```bash
   python Self_Learning_Clinical_NLP_Tutorial.py
   ```

The script writes aggregate summaries and figures to `self_learning_outputs/`. It does not export raw note text.

## Reproducibility

The tutorial fixes the random seed at 42. Word2Vec uses a 50-dimensional skip-gram model, a window of 5, and 150 epochs. t-SNE evaluates perplexities of 5, 10, 20, and 30, then chooses the setting with the strongest local-neighborhood trustworthiness for each pipeline.

## Important limitation

This is a transparent learning workflow, not a clinical decision-support tool or a benchmark NER evaluation. The entity vocabulary is intentionally controlled for comparison; a production study should use annotated data and report validated precision, recall, and downstream performance.

## Libraries and data citation

- Johnson AEW, et al. MIMIC-III, a freely accessible critical care database. *Scientific Data*. 2016;3:160035. doi:10.1038/sdata.2016.35
- spaCy, scispaCy, medSpaCy, gensim, and scikit-learn
