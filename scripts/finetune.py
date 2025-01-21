from pathlib import Path
from typing import Annotated, Dict, List
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import typer
from typer import Option

from llmpipe import read_data


def compute_metrics(pred: EvalPrediction) -> Dict:
    """Compute classification metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Get detailed classification metrics
    report = classification_report(labels, preds, output_dict=True)
    conf_matrix = confusion_matrix(labels, preds)
    
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": conf_matrix.tolist()
    }
    
    # Add per-class metrics
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                if label not in ["macro avg", "weighted avg"]:
                    metrics[f"{label}_{metric}"] = value
                    
    return metrics


def finetune(
    model_path: Annotated[str, Option(help="Local or HuggingFace model path")] = "roberta-base",
    train_input_data_path: Annotated[str, Option(help="Path to training data")] = None,
    val_input_data_path: Annotated[str, Option(help="Path to validation data")] = None,
    test_input_data_path: Annotated[str, Option(help="Path to test data")] = None,
    output_path: Annotated[str, Option(help="Path to save model and metrics")] = None,
    num_epochs: Annotated[int, Option(help="Number of training epochs (0 to skip training)")] = 0,
    learning_rate: Annotated[float, Option(help="Learning rate")] = 0.00001,
    batch_size: Annotated[int, Option(help="Batch size for training and evaluation")] = 8,
):
    """Fine-tune a RoBERTa model for classification using the label field."""
    
    # Expand user paths
    train_input_data_path = str(Path(train_input_data_path).expanduser())
    val_input_data_path = str(Path(val_input_data_path).expanduser())
    test_input_data_path = str(Path(test_input_data_path).expanduser())
    output_path = str(Path(output_path).expanduser())
    
    # Load datasets
    train_data = read_data(train_input_data_path)
    val_data = read_data(val_input_data_path)
    test_data = read_data(test_input_data_path)
    
    # Get unique labels and create label mapping
    all_labels = sorted(list(set([d["label"] for d in train_data + val_data + test_data])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(all_labels)
    
    # Convert to HF datasets
    def convert_to_dataset(data: List[Dict]) -> Dataset:
        return Dataset.from_dict({
            "text": [d["dialog"] for d in data],
            "label": [label2id[d["label"]] for d in data]
        })
    
    train_dataset = convert_to_dataset(train_data)
    val_dataset = convert_to_dataset(val_data)
    test_dataset = convert_to_dataset(test_data)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model if epochs > 0
    if num_epochs > 0:
        trainer.train()
    
    # Evaluate on validation and test sets
    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(test_dataset)
    
    # Save metrics and label mappings
    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "label_mappings": {
            "label2id": label2id,
            "id2label": id2label
        }
    }
    
    metrics_path = Path(output_path) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(finetune)
    app()
