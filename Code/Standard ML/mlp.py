import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from Code.Dataprocessing import df_processing as dp
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split, KFold, \
    StratifiedShuffleSplit, GridSearchCV

# Read with finBERT labelled csv file
file = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\ar_labelled_market.csv"
df = pd.read_csv(file, index_col=0)

########################################################################################################################
# Data preprocessing
########################################################################################################################

# Convert finBERT labels into categorical variable
df["finBERT"] = pd.Categorical(df["finBERT"], ordered=True, categories=["negative", "neutral", "positive"])

# Add relative return
df = dp.add_relative_return(df, 'sp_10_pct', 'sp_90_pct')

# Only take data for only specific stock
# df = df.loc[df["stock"] == "MRK"]
df.sort_values(by=["stock", "date"], inplace=True)

# Apply balancing to dataset
df = dp.balance_dataframe(df, 'finBERT')
df.sort_values(by=["stock", "date"], inplace=True)

# Apply windowing
dp.get_window_data(df, {'open': "mean", 'high': "mean", 'low': "mean", 'close': "mean", 'adj close': "mean",
                        'volume': "mean", 'return': "mean", 'log_return': "sum", 'sp_mean': "mean", 'sp_var': "mean",
                        'sp_10_pct': "mean", 'sp_25_pct': "mean", 'sp_50_pct': "mean", 'sp_75_pct': "mean",
                        'sp_90_pct': "mean", 'under': "median", 'neutral': "median", 'out': "median"}, 3)

df = df.sample(n=50)
df["finBERT"].value_counts().plot(kind="bar")
plt.show()

########################################################################################################################
# Define independent variables
########################################################################################################################

# Define independent and dependent variable
independent1 = ['open', 'high', 'low', 'close', 'adj close', 'volume', 'return', 'log_return', 'sp_mean',
                'sp_var', 'sp_10_pct', 'sp_25_pct', 'sp_50_pct', 'sp_75_pct', 'sp_90_pct', "under", "neutral", "out"]
independent2 = ['return', 'sp_var', 'sp_25_pct', 'sp_75_pct']
independent3 = ['return', 'sp_var', "under", "neutral", "out"]
independent4 = ['return', 'sp_var']
independent5 = ["under", "neutral", "out"]
X = df.loc[:, df.columns.isin(independent1)].values
y = df.loc[:, df.columns == "finBERT"].values.ravel()

########################################################################################################################
# Create train/test split and standardize
########################################################################################################################

# Create train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Standardize independent variable - use this if there are categorical independent variables
ct = ColumnTransformer(
    transformers=[
        ('standardization', StandardScaler(), list(range(0, 16)))
    ],
    remainder='passthrough'
)
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

########################################################################################################################
# MLP - Hyper Parameter Tuning
########################################################################################################################

# Define model
model = MLPClassifier(early_stopping=True, validation_fraction=0.2, n_iter_no_change=25)

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(150, 100, 50, 30), (120, 80, 40, 9), (100, 50, 30, 3)],
    'max_iter': [200, 300, 500],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

grid = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, verbose=3)
grid.fit(X_train, y_train)

print("The best hyper parameters are:")
print(grid.best_params_)

########################################################################################################################
# MLP - Make predictions and get confusion matrix
########################################################################################################################

# Define model
model = MLPClassifier()
model.set_params(**grid.best_params_)

# Fit model
model.fit(X_train, y_train)

# Make predictions
y_hat = model.predict(X_test)

# Print loss curve
plt.plot(model.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Get classification report
print(classification_report(y_test, y_hat, target_names=["negative", "neutral", "positive"]))

# Get confusion matrix
cm = confusion_matrix(y_test, y_hat, labels=["negative", "neutral", "positive"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
