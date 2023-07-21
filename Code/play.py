import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_labelled_market_gpt35.csv")


print(classification_report(df["finBERT"], df["sentiment"], target_names=["negative", "neutral", "positive"]))

cm = confusion_matrix(df["finBERT"], df["sentiment"], labels=["negative", "neutral", "positive"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "neutral", "positive"])
disp.plot()
plt.show()
