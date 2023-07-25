import numpy as np
from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

# Load data
labels = ClassLabel(num_classes=3, names=["negative", "neutral", "positive"])

# Load dataset
dataset = load_dataset("csv", data_files="/content/drive/MyDrive/masterProject/av_train.csv")
dataset = dataset.rename_column("finBERT", "label")
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
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Set hyperparameters
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
training_args = TrainingArguments(
    output_dir="distilBERT_finetuning",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)


# Evaluate model
def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

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
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune model
trainer.train()

# Evaluate model
print(trainer.evaluate())

# Save model
trainer.save_model("/content/drive/MyDrive/masterProject/")
