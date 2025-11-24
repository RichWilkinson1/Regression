# Customer Spend Model

A simple project demonstrating structured Python modelling outside a notebook.

###############################################

1) Create the project folder

On your machine (e.g., Desktop), create a folder named:

customer_spend_model

2) Open the folder in VS Code

Open VS Code → File → Open Folder → select customer_spend_model.

3) Open a terminal inside VS Code

VS Code menu: View → Terminal.

4) Create a virtual environment

python -m venv .venv


5) Activate the virtual environment

PowerShell:

.venv\Scripts\activate


6) Install core Python libraries

pip install pandas scikit-learn matplotlib


7) Create a requirements.txt file

After installing packages in a virtual environment, I freeze them into a requirements.txt file so the project can be recreated exactly on other machines or in production

pip freeze > requirements.txt


8) Create project folders
In VS Code’s sidebar, create folders:

src/
notebooks/
tests/


9) Create your Python package folder
Inside src/, create:

spend_model/


10) Create an initializer file
Inside src/spend_model/, create:

__init__.py

# __init__.py makes a folder importable as a Python package.

11) Create your reusable code modules
Inside src/spend_model/, create files:

data_load.py
features.py
train.py


12) Create a notebook for exploration
Inside notebooks/, create:

01_exploration.ipynb


13) Create a test file
Inside tests/, create:

test_features.py


14) Create a README
In the root folder, create:

README.md


Your structure should now look like:

customer_spend_model/
├── .venv/
├── src/
│   └── spend_model/
│       ├── __init__.py
│       ├── data_load.py
│       ├── features.py
│       └── train.py
├── notebooks/
│   └── 01_exploration.ipynb
├── tests/
│   └── test_features.py
├── requirements.txt
└── README.md

###############################################

Linear regression relies on five key assumptions:

1) Linearity
It assumes the relationship between predictors and the target is linear.
If the true relationship is curved or non-linear, the model will miss important patterns.
We check this using scatter plots or residual vs fitted plots.

2) Independence of errors
Residuals must not be correlated with each other.
This matters in time series or clustered data – otherwise the model overstates confidence.
We’d check this with autocorrelation (e.g., Durbin-Watson test).

3) Homoscedasticity (constant variance of errors)
The spread of the residuals should be roughly the same across all levels of prediction.
If variance grows or shrinks, the model is biased for certain value ranges.
We check this visually with residual plots.

4) Normality of residuals
Residuals should be roughly normally distributed.
This affects the reliability of confidence intervals and p-values.
We check using histograms, Q-Q plots, or normality tests.

5) No multicollinearity
Predictors shouldn’t be highly correlated with each other.
Collinearity destabilises coefficients and makes interpretation unreliable.
We detect this using correlation matrices or VIF (Variance Inflation Factor).

Residuals are the differences between actual and predicted values, and analysing them helps us check model validity and assumptions.

###############################################

R² (Coefficient of Determination)
- Proportion of variance in the target explained by the model
- How much of the outcome can be explained by the features. Higher is better.

Adjusted R²
- R² adjusted for the number of predictors
- Penalises adding meaningless variables. Helps judge whether adding new features really improves the model.

MAE (Mean Absolute Error)
- Average of absolute differences between predictions and actuals
- On average, how far off your predictions are in the same units as the target (e.g., dollars).

MSE (Mean Squared Error)
- Average of squared differences between predictions and actuals
- Punishes big mistakes more heavily than small ones.

RMSE (Root Mean Squared Error)
- Square root of MSE
- Magnitude of error, in the same units as the target. Shows typical size of mistakes.