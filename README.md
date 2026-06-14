# Output
<img width="516" height="315" alt="image" src="https://github.com/user-attachments/assets/1cfb7a11-d1d5-446b-92ad-3d488e2a09c1" />


# Inflation Rate Prediction in Pakistan Using Machine Learning

This project predicts Pakistan's inflation rate using historical economic indicators and a Lasso Regression model. It was prepared as an Artificial Intelligence lab project and converted from the original Word report into a runnable GitHub project.

## Project Overview

The model uses economic indicators such as manufacturing output, foreign direct investment, food imports, exchange rate, and military expenditure to predict annual consumer price inflation.

Lasso Regression was selected because it can predict a numerical target while also reducing the effect of less useful features through regularization.

## Dataset

The dataset is included at:

```text
data/inflation_data.csv
```

An Excel copy is also available at `data/inflation_data.xlsx`.

Columns used by the model:

- Inflation, consumer prices (annual %) - target variable
- Manufacturing, value added (% of GDP)
- Foreign direct investment, net inflows (% of GDP)
- Food imports (% of merchandise imports)
- Official exchange rate (LCU per US$, period average)
- Military expenditure (% of GDP)

## How to Run

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model and enter your own values:

```bash
python inflation_prediction.py
```

4. Run a quick sample prediction:

```bash
python inflation_prediction.py --sample
```

## Methodology

The workflow follows these steps:

1. Load the Excel dataset.
2. Remove missing values.
3. Select inflation as the dependent variable.
4. Select economic indicators as independent variables.
5. Standardize features using `StandardScaler`.
6. Split data into training and testing sets.
7. Train a `LassoCV` regression model with 5-fold cross-validation.
8. Evaluate model performance using Mean Squared Error and R2 score.
9. Predict inflation from user-provided values.

## Project Files

- `inflation_prediction.py` - main Python script
- `data/inflation_data.csv` - main dataset used by the script
- `data/inflation_data.xlsx` - Excel copy of the dataset
- `docs/ai inflation final.docx` - original project report
- `assets/` - images extracted from the report

## Course

Artificial Intelligence Lab Project-II  
Shaheed Zulfikar Ali Bhutto Institute of Science & Technology
