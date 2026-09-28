# Wine Quality Classification: SVC with Hyperparameter Tuning

Classifies Portuguese "Vinho Verde" wines as lower or higher quality from their chemical properties using a Support Vector Classifier, then tunes the model with grid search and checks whether tuning helps.

## What I did
- Loaded the Wine Quality dataset (6,497 red and white wines) and inspected missing values and the quality distribution
- Converted the quality score to a binary label (4 or below = lower, 5 or above = higher), filled missing values, standardised the features and made a 75/25 train-test split
- Trained a baseline SVC and defined a custom log-loss scoring function
- Ran a 5-fold cross-validated grid search over `C` and `gamma`
- Compared the default and tuned models against a naive baseline, using accuracy and confusion matrices

## Results
| Model | Accuracy (test set) | Wrong predictions |
|---|---|---|
| Naive baseline (always "higher quality") | 0.9643 | 58 |
| SVC default | 0.9637 | 59 |
| SVC tuned (C=1, gamma=1) | 0.9649 | 57 |

Both SVC models score about 96.4-96.5%, but so does a model that labels every wine "higher quality", because only about 4% of the wines are low quality. The default model finds 0 of the 58 low-quality wines in the test set and the tuned model finds 1, so tuning made almost no practical difference. The main lesson is that accuracy alone is misleading on imbalanced data.

Next steps: handle the class imbalance (`class_weight='balanced'` or resampling), use recall/precision/F1 for the low-quality class, score log loss on probabilities, widen the grid, and put scaling inside a `Pipeline` to avoid leakage.

## Files
| File | Purpose |
|---|---|
| `Wine_Quality_SVC_Tuning.ipynb` | The full analysis |
| `requirements.txt` | Python packages needed |

## How to run
```
pip install -r requirements.txt
jupyter notebook Wine_Quality_SVC_Tuning.ipynb
```
Run the notebook top to bottom. An internet connection is needed because the data is read from the web.

## Data source
Wine Quality dataset, UCI Machine Learning Repository, provided through ExploreAI Academy coursework.
