import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import StandardScaler
from Code.Dataprocessing import df_processing as dp
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from mixed_naive_bayes import MixedNB

# https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

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
# df = df.loc[df["stock"] == "MS"]
df.sort_values(by=["stock", "date"], inplace=True)

# Apply balancing to dataset
df = dp.balance_via_undersampling(df, 'finBERT')
df.sort_values(by=["stock", "date"], inplace=True)

# Show the distribution of positive, neutral and negative labels
df["finBERT"].value_counts().plot(kind="bar")
plt.show()

# Apply windowing
dp.get_window_data(df, {'open': "mean", 'high': "mean", 'low': "mean", 'close': "mean", 'adj close': "mean",
                        'volume': "mean", 'return': "mean", 'log_return': "sum", 'sp_mean': "mean", 'sp_var': "mean",
                        'sp_10_pct': "mean", 'sp_25_pct': "mean", 'sp_50_pct': "mean", 'sp_75_pct': "mean",
                        'sp_90_pct': "mean", 'under': "median", 'neutral': "median", 'out': "median"}, 3)


########################################################################################################################
# Define independent variables
########################################################################################################################

# Define independent and dependent variable
independent1 = ['open', 'high', 'low', 'close', 'adj close', 'volume', 'return', 'log_return', 'sp_mean',
                'sp_var', 'sp_10_pct', 'sp_25_pct', 'sp_50_pct', 'sp_75_pct', 'sp_90_pct']
independent2 = ['return', 'sp_var', 'sp_25_pct', 'sp_75_pct']
independent3 = ['return', 'sp_var', "under", "neutral", "out"]
independent4 = ['return', 'sp_var']
independent5 = ["under", "neutral", "out"]
X = df.loc[:, df.columns.isin(independent5)].values
y = df.loc[:, df.columns == "finBERT"].values.ravel()

########################################################################################################################
# Naive Bayes
########################################################################################################################

# Create train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Define categorical naive bayes
model = CategoricalNB(force_alpha=True)
# model = GaussianNB()
# model = MixedNB(categorical_features=[2, 3, 4])
model.fit(X_train, y_train)

# Make predictions
y_hat = model.predict(X_test)

# Get classification report
print(classification_report(y_test, y_hat, target_names=["negative", "neutral", "positive"]))

# Get confusion matrix
cm = confusion_matrix(y_test, y_hat, labels=["negative", "neutral", "positive"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
