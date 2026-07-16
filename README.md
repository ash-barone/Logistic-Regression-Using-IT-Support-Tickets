# Logistic-Regression-Using-IT-Support-Tickets
A machine learning capstone that builds a logistic regression model that predicts which IT support tickets will breach their Service Level Agreement (SLA) *before* they breach, using only information available the moment a ticket is created. This lets support teams flag at-risk tickets early and allocate resources proactively instead of reacting after the fact.

## Research Question

Can a model predict SLA breaches from ticket attributes known at creation time,
accurately enough (>70%) to be useful for proactive triage?

## Dataset

10,000 IT support tickets from a publicly available Kaggle dataset, with a 36.6%
breach rate. The data file (`support_tickets_10000.csv`) is not included here;
download it from Kaggle and place it in the project root before running.

## Approach

- **Leakage-aware feature selection:** used only the six attributes known at ticket
  creation (customer segment, product area, priority, channel, agent tenure,
  ticket sentiment). Post-resolution fields like resolution time and escalation
  status were deliberately excluded so the model cannot "cheat."
- **Preprocessing:** log-transformed skewed agent tenure, one-hot encoded
  categoricals, and confirmed low multicollinearity with a VIF check.
- **Two models compared:** a cross-validated Logistic Regression (interpretable,
  gives direction and size of each effect) and an XGBoost classifier (a stronger
  learner used to validate the findings from a different angle).
- **Validation:** stratified 80/20 split, an ordinal-vs-one-hot encoding
  comparison, a sentiment-leakage sensitivity check, and statsmodels p-values to
  confirm which predictors are statistically significant.

## Results

- **Logistic Regression: 87.3% accuracy. XGBoost: 86.7%.** Both far exceeded the
  70% goal and agreed closely, which strengthens confidence in the findings.
- Roughly 85% precision on flagged tickets (few false alarms) and about 80% of
  actual breaches caught early.
- **Priority is by far the dominant predictor** (about 72% of model importance).
  Urgent tickets carry roughly 11x the odds of a breach; low priority and positive
  sentiment sharply lower it. Channel had almost no effect.

## Business Value

Support teams can flag high-risk tickets the moment they arrive and focus effort
where it matters, rather than treating every ticket equally or reacting only to
incoming priority flags.

## Limitations

Single-source data that may not transfer to other organizations, no customer
demographics, Logistic Regression's linearity assumption, and a default 0.50
flagging threshold that a real deployment would tune to team capacity.

## Repository Contents

- `Logistic-Regression-IT-Support-Tickets.ipynb` — full analysis notebook
- `Data-Analytics-Report.docx` — detailed written report
- `Executive-Summary.docx` — one-page summary for non-technical stakeholders
- `Presentation-Slides.pdf` — slide deck presenting the project

## Tech Stack

Python, pandas, scikit-learn, XGBoost, statsmodels, matplotlib, seaborn

## Running It

1. Download the dataset and place `support_tickets_10000.csv` in the project root.
2. Install dependencies: `pip install pandas scikit-learn xgboost statsmodels matplotlib seaborn scipy`
3. Open the notebook and run all cells.

## Updates

- **2026-07:** Modernized for current XGBoost (removed the obsolete
  use_label_encoder argument) and corrected a mislabeled chart title.
