# Synthetic priority analysis

This is a pipeline diagnostic on the checked-in synthetic dataset. It is not a
VCET performance estimate. The baseline was retrained with `make_splits(seed=42)`:
922 training rows, 299 validation rows and 338 test rows. Duplicate references
and exact normalized-text groups are kept within one split. The confidence gate
was selected on validation data only; the test split was used only for the
reported final evaluation.

## Held-out test observations

The test split contains 338 examples. True priorities are distributed as:

| Priority | Count |
| --- | ---: |
| 1 | 35 |
| 2 | 147 |
| 3 | 128 |
| 4 | 27 |
| 5 | 1 |

For the high-priority definition `priority >= 4`, only 28 of 338 examples are
positive (8.3%). The shipped rounded regression output predicted three examples
as high priority: recall was 7.1%, precision 66.7%, and raw regression MAE was
0.574. Applying a raw score cutoff of 4.0 found none of the high-priority test
examples. That raw-cutoff recall is reported separately because the application
returns a rounded, clipped 1–5 priority value.

The validation split had 29 high-priority examples. At the validation-selected
operating threshold, the rounded output had 27.6% high-priority recall and 100%
precision; a raw score cutoff of 4.0 again had 0% recall. These measures are
unstable with so few high-priority examples, especially the single priority-5
test record.

## Decision

Keep the existing ordinal regression approach for now. Ridge in the local
baseline and Huber loss in the transformer preserve the ordering between
priority levels. The synthetic data has few priority-4/5 examples, and the
current evidence does not justify switching the model or reweighting its loss.
The application should continue to treat low confidence as a reason for human
review; priority predictions should not independently trigger an irreversible
workflow action.

Training reports now distinguish recall of the actual rounded application
decision from recall at a raw-score cutoff. Future model comparisons should
choose any operating threshold using validation data, then report once on the
untouched test split. Do not tune from these synthetic results for institutional
use.
