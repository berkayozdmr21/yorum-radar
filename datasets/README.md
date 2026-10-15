# Dataset provenance

All examples in this directory are manually authored synthetic Turkish reviews. No customer data was scraped or copied from an external dataset.

- `train.csv`: 120 labeled training examples, 40 per class.
- `holdout.csv`: 36 separately authored examples, 12 per class. Exact normalized overlap with training is rejected by the training script.
- `demo.csv`: the same 36 holdout texts with synthetic dates and ratings, for dashboard demonstration. Dashboard predictions on this demo are **not an independent evaluation** beyond the holdout report.

Labels: `positive`, `negative`, `neutral`. UTF-8 with BOM. The set is small, curated and not representative of real Turkish customer feedback. Its metrics cannot support production accuracy claims. Near-duplicates, shared author/style and topic vocabulary may make evaluation optimistic even without exact overlap.

For real evaluation, obtain appropriately licensed, independently labeled data, split by source/product/user where relevant, and retain a final untouched test set. Rating alone is not a reliable sentiment label. Never report synthetic results as real customer performance.
