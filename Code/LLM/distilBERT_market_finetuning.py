import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

from Code.Dataprocessing import df_processing as dp

df = pd.read_csv(r"C:\Users\chris\IdeaProjects\masterProject\Dataset\av_train.csv")

df = dp.add_relative_return_ordinal(df, "sp_20_pct", "sp_80_pct")

df = dp.balance_via_oversampling(df, "relative_return")

print(df.value_counts(subset="relative_return"))

# Load data
labels = ClassLabel(num_classes=3, names=["negative", "neutral", "positive"])
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# Load dataset
dataset = load_dataset(df)
dataset = dataset.rename_column("relative_return", "label")
dataset = dataset.rename_column("summary", "text")

# Split into 80% training and 20% validation
dataset = dataset["train"].train_test_split(train_size=0.8)

# Tokenize dataset using distilbert-base-uncased
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    examples["label"] = labels.str2int(examples["label"])
    return tokenizer(examples["text"], padding=True, truncation=True)


tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# Convert to PyTorch tensors for faster training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model and specify the number of labels
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", id2label=id2label,
                                                           label2id=label2id, num_labels=3)

# Set hyperparameters
training_args = TrainingArguments(
    output_dir="distilBERT_finetuning",
    learning_rate=3e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=1,
    weight_decay=0,
    evaluation_strategy="steps",
    # push_to_hub=True,
    eval_steps=10,
    max_steps=100,
    save_steps=0,
    logging_steps=1
)


# Evaluate model
def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]

    return {"accuracy": accuracy, "f1": f1}


# Create trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune model
trainer.train()

# Evaluate model
trainer.evaluate()

# Upload the model to the Hub
# trainer.push_to_hub()
