# Yorum Radar — Baseline Model Card

## Intended use

Educational portfolio, local CSV exploration, and a reproducible baseline for Turkish sentiment classification. Labels: negative, neutral, positive. Not validated for automated decisions about customers, employees, refunds or service eligibility.

## Data and provenance

120 authored synthetic training examples (40 per class), 36 separately authored holdout examples (12 per class). No external customer dataset, scraped reviews or paid API was used. The demo CSV reuses the holdout texts with invented dates and ratings. It is a visualization fixture, not a third independent dataset.

Normalization removes whitespace variation and handles Turkish I/İ. Exact duplicate normalized texts are deduplicated before splitting. Conflicting labels and exact training/test overlap fail training. Semantic/near-duplicate leakage is not automatically detected. Shared author and topic vocabulary can inflate performance.

## Model and training

FeatureUnion: word TF-IDF (1,2) + character-within-word TF-IDF (3,5). Sublinear term frequency. Logistic Regression, C=4, balanced class weights, max_iter=1500, seed=42. Comparison model: Multinomial Naive Bayes, alpha=0.5. The active baseline is chosen in advance, not selected by test performance.

The bundled holdout macro-F1 is approximately 0.945 for Logistic Regression and 0.918 for Naive Bayes. See `ml/baseline_report.json` for exact per-class results, confusion matrices and errors. These metrics describe a small synthetic set only.

## Outputs and interpretation

The predicted sentiment is the highest-probability class. Probabilities are not calibrated. The UI uses 0.55 as an illustrative review threshold, not a validated operational cutoff. Keyword explanations display positive word-feature contributions to the chosen class and omit character features; they are partial feature attributions.

Topic labels come from substring keyword rules in `backend/topics.py`, not the sentiment classifier. A review can have multiple topics. Topic counts overlap; sentiment is currently one label per review.

## Known limitations

Negation, sarcasm, mixed sentiment, informal spelling, dialects, emerging products and domain shift can reduce accuracy. Dataset size is insufficient to establish subgroup fairness or production readiness. Ratings are not used as model inputs. Date and rating cells that cannot be parsed are counted and left empty.

## Privacy and deployment

Local inference; no review text sent to an external model API. SQLite and locally trained artifacts are ignored by Git. Optional transformer training downloads a foundation model but does not upload training text. Dependencies and optional web fonts require network access when first fetched.

Use independent licensed data, label quality audits, source-aware splits, calibration, error inspection and human review before operational deployment.
