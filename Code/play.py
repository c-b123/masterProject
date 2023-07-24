import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_raw.csv")


# df = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\financial_phrasebank_allagree_gpt35.csv")


# print(classification_report(df["label"], df["gpt-3.5-turbo"], target_names=["negative", "neutral", "positive"]))

# cm = confusion_matrix(df["label"], df["gpt-3.5-turbo"], labels=["negative", "neutral", "positive"])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
# disp.plot()
# plt.show()
