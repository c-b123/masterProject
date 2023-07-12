import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

# https://machinelearningmastery.com/multinomial-logistic-regression-with-python/

# Read with finBERT labelled csv file
file = r"C:\Users\chris\IdeaProjects\masterProject\Dataset\analyst_ratings_labelled.csv"
df = pd.read_csv(file, index_col=0)

# Define independent and dependent variable
independent = ["open", "high", "low", "close", "adj close", "volume", "return", "log_return"]
X = df.loc[:, df.columns.isin(independent)]
y = df.loc[:, df.columns == "finBERT"]


########################################################################################################################
# Data exploration
########################################################################################################################

# Show the distribution of positive, neutral and negative labels
df["finBERT"].value_counts().plot(kind="bar")
plt.show()


########################################################################################################################
# Multinomial Logistic Regression
########################################################################################################################

# Define Multinomial Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Evaluate the model and collect the scores
n_scores = cross_val_score(model, X, np.ravel(y), scoring='accuracy', cv=cv, n_jobs=-1)

# Report performance
print('Mean Accuracy for Multinomial Logistic Regression: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


########################################################################################################################
# Multinomial Logistic Regression with regularization
########################################################################################################################

# List of models to evaluate
def get_models():
    models = dict()
    for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
        # Create name for model
        key = '%.4f' % p

        # Turn off penalty in some cases
        if p == 0.0:
            # No penalty in this case
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty=None)
        else:
            models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)

    return models


# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

    return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model and collect the scores
    scores = evaluate_model(model, X, np.ravel(y))
    # store the results
    results.append(scores)
    names.append(name)
    # summarize progress along the way
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


########################################################################################################################
# Ordinal Logistic Regression
########################################################################################################################


########################################################################################################################
# Ordinal Logistic Regression with regularization
########################################################################################################################


########################################################################################################################
# Further ideas
########################################################################################################################

# add more KPIs with Pyfolio
# only positive and negative
