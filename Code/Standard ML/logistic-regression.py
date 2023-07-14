import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Read with finBERT labelled csv file
file = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\ar_labelled_market.csv"
df = pd.read_csv(file, index_col=0)

# Convert finBERT labels into categorical variable
df["finBERT"] = pd.Categorical(df["finBERT"], ordered=True, categories=["negative", "neutral", "positive"])

# Only take data for only specific stock
# print(df["stock"].value_counts())
# df = df.loc[df["stock"] == "MRK"]


########################################################################################################################
# Balance dataset and define independent variables
########################################################################################################################


def balance_dataframe(df, target_column):
    class_counts = df[target_column].value_counts()
    min_class_count = class_counts.min()
    balanced_df = df.groupby(target_column).apply(lambda x: x.sample(min_class_count, random_state=42))
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


# Apply balancing to dataset
df = balance_dataframe(df, 'finBERT')

# Show the distribution of positive, neutral and negative labels
df["finBERT"].value_counts().plot(kind="bar")
plt.show()

# Define independent and dependent variable
independent1 = ['open', 'high', 'low', 'close', 'adj close', 'volume', 'return', 'log_return', 'sp_mean',
                'sp_var', 'sp_10_pct', 'sp_25_pct', 'sp_50_pct', 'sp_75_pct', 'sp_90_pct']
independent2 = ['return', 'sp_var', 'sp_25_pct', 'sp_75_pct']
X = df.loc[:, df.columns.isin(independent2)].values
y = df.loc[:, df.columns == "finBERT"].values.ravel()


########################################################################################################################
# Multinomial Logistic Regression
########################################################################################################################

# Create train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Standardize independent variable
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

# Define Multinomial Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_hat = model.predict(X_test)

# Get classification report
print(classification_report(y_test, y_hat, target_names=["negative", "neutral", "positive"]))


########################################################################################################################
# Multinomial Logistic Regression with KFold validation
########################################################################################################################

# Standardize independent variable
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define Multinomial Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluate the model and collect the scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# Report performance
print('Accuracy for Multinomial Logistic Regression:\n'
      '{0:.3f} ({1:.3f})'.format(np.mean(n_scores), np.std(n_scores)))


########################################################################################################################
# Multinomial Logistic Regression with regularization
########################################################################################################################

# List of models to evaluate
def get_models():
    models = dict()
    for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        # Create name for model
        key = '%.4f' % p

        if p == 0.0:
            # No penalty in this case
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty=None)
        else:
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)

    return models


# Evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    return scores


# Get the models to evaluate
models = get_models()

# Evaluate the models and store results
results, names = list(), list()
print("Accuracy for Multinomial Logistic Regression with regularization:")
for name, model in models.items():
    # Evaluate the model and collect the scores
    scores = evaluate_model(model, X, y)
    # Store the results
    results.append(scores)
    names.append(name)
    # Summarize progress along the way
    print('>{0} {1:.3f} ({2:.3f})'.format(name, np.mean(scores), np.std(scores)))

########################################################################################################################
# Further ideas
########################################################################################################################

# add more KPIs with Pyfolio
# add S&P500 return as KPI -done
# only positive and negative
# balance dataset - done
# try ordinal logistic regression
# lag data
# Standardizing/Normalize - done
# use window of returns for independent variable
