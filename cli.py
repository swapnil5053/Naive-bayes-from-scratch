"""
Command-line interface for training and inference with the custom Naive Bayes classifier.

Provides functionality for model training, evaluation, and prediction with explainability.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import box
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.naive_bayes import ExplainableNaiveBayes

console = Console()

# File paths
DATA_DIR = Path(__file__).parent
MODEL_PATH = DATA_DIR / "model.pkl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"


def load_pubmed_data(filepath: Path) -> pd.DataFrame:
    """Load PubMed RCT dataset from text file."""
    labels, sentences = [], []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            label, sent = line.split('\t', maxsplit=1)
            labels.append(label)
            sentences.append(sent)
    
    return pd.DataFrame({'label': labels, 'sentence': sentences})


def train_model():
    """Train the Naive Bayes classifier and save it."""
    console.print("\n[bold]Training Naive Bayes Classifier[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Load data
        task = progress.add_task("[cyan]Loading PubMed 20k RCT dataset...", total=None)
        train_df = load_pubmed_data(DATA_DIR / "train.txt")
        test_df = load_pubmed_data(DATA_DIR / "test.txt")
        progress.update(task, completed=True)
        
        console.print(f"Loaded {len(train_df):,} training samples")
        console.print(f"Loaded {len(test_df):,} test samples\n")
        
        # Extract features
        task = progress.add_task("[cyan]Extracting features with CountVectorizer...", total=None)
        vectorizer = CountVectorizer(
            lowercase=True,
            strip_accents='unicode',
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_features=50000
        )
        
        X_train = vectorizer.fit_transform(train_df['sentence'])
        X_test = vectorizer.transform(test_df['sentence'])
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        progress.update(task, completed=True)
        
        console.print(f"Vocabulary size: {X_train.shape[1]:,} features\n")
        
        # Train model
        task = progress.add_task("[cyan]Training custom Naive Bayes classifier...", total=None)
        model = ExplainableNaiveBayes(alpha=1.0)
        model.fit(X_train, y_train, feature_names=vectorizer.get_feature_names_out())
        progress.update(task, completed=True)
        
        console.print("Training complete\n")
        
        # Evaluate
        task = progress.add_task("[cyan]Evaluating on test set...", total=None)
        y_pred = model.predict(X_test)
        progress.update(task, completed=True)
    
    # Display results
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    console.print(Panel.fit(
        f"[bold green]Accuracy:[/bold green] {accuracy:.4f}\n"
        f"[bold green]Macro F1 Score:[/bold green] {f1_macro:.4f}",
        title="Overall Performance",
        border_style="green"
    ))
    
    # Classification report table
    report = classification_report(y_test, y_pred, output_dict=True)
    
    table = Table(title="\nDetailed Classification Report", box=box.ROUNDED)
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Precision", justify="right", style="magenta")
    table.add_column("Recall", justify="right", style="yellow")
    table.add_column("F1-Score", justify="right", style="green")
    table.add_column("Support", justify="right", style="blue")
    
    classes = sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
    
    for class_name in classes:
        metrics = report[class_name]
        table.add_row(
            class_name,
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}",
            str(int(metrics['support']))
        )
    
    # Add macro avg
    table.add_section()
    macro = report['macro avg']
    table.add_row(
        "[bold]Macro Avg[/bold]",
        f"[bold]{macro['precision']:.3f}[/bold]",
        f"[bold]{macro['recall']:.3f}[/bold]",
        f"[bold]{macro['f1-score']:.3f}[/bold]",
        str(int(macro['support']))
    )
    
    console.print(table)
    
    # Show top features per class
    console.print("\n[bold cyan]Top Indicative Words Per Class[/bold cyan]\n")
    top_features = model.get_top_features_per_class(n=8)
    
    feature_table = Table(box=box.SIMPLE)
    for class_name in sorted(top_features.keys()):
        feature_table.add_column(class_name, style="cyan")
    
    max_rows = max(len(words) for words in top_features.values())
    for i in range(max_rows):
        row = []
        for class_name in sorted(top_features.keys()):
            words = top_features[class_name]
            if i < len(words):
                word, score = words[i]
                row.append(f"{word}")
            else:
                row.append("")
        feature_table.add_row(*row)
    
    console.print(feature_table)
    
    # Save model
    console.print("\n[cyan]Saving model and vectorizer...[/cyan]")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    console.print(f"[green]Model saved to {MODEL_PATH}[/green]")
    console.print(f"[green]Vectorizer saved to {VECTORIZER_PATH}[/green]\n")


def predict_text(text: str):
    """Predict class for input text with explainability."""
    # Load model and vectorizer
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        console.print("[red]Error: Model not found. Please run 'python cli.py train' first.[/red]")
        return
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Get explanation
    explanation = model.explain_prediction(text, vectorizer, top_n=5)
    
    # Display input text
    console.print(Panel(
        text,
        title="Input Text",
        border_style="blue"
    ))
    
    # Display prediction
    predicted_class = explanation['predicted_class']
    console.print(Panel.fit(
        f"[bold green]{predicted_class}[/bold green]",
        title="Predicted Class",
        border_style="green"
    ))
    
    # Display probability distribution
    console.print("\n[bold cyan]Probability Distribution[/bold cyan]\n")
    
    probs = explanation['probabilities']
    max_prob = max(probs.values())
    
    for class_name in sorted(probs.keys()):
        prob = probs[class_name]
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        
        style = "green" if class_name == predicted_class else "white"
        console.print(f"[{style}]{class_name:12s}[/{style}] {bar} {prob:.4f}")
    
    # Display explainability
    console.print("\n[bold cyan]Explainability: Top Contributing Words[/bold cyan]\n")
    
    if explanation['top_words']:
        explain_table = Table(box=box.ROUNDED)
        explain_table.add_column("Rank", style="cyan", justify="center")
        explain_table.add_column("Word", style="magenta")
        explain_table.add_column("Count", justify="right", style="yellow")
        explain_table.add_column("Contribution", justify="right", style="green")
        
        for rank, (word, contribution, count) in enumerate(explanation['top_words'], 1):
            explain_table.add_row(
                str(rank),
                word,
                str(int(count)),
                f"{contribution:.4f}"
            )
        
        console.print(explain_table)
        
        console.print(f"\n[dim]These words had the highest impact on classifying this text as '{predicted_class}'.[/dim]\n")
    else:
        console.print("[yellow]No significant words found in the input text.[/yellow]\n")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/red]")
        console.print("  [cyan]python cli.py train[/cyan]                    - Train the model")
        console.print("  [cyan]python cli.py predict \"Your text here\"[/cyan]  - Predict class for text")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
        train_model()
    elif command == "predict":
        if len(sys.argv) < 3:
            console.print("[red]Error: Please provide text to classify.[/red]")
            console.print("[cyan]Example: python cli.py predict \"Your medical text here\"[/cyan]")
            sys.exit(1)
        text = " ".join(sys.argv[2:])
        predict_text(text)
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[cyan]Available commands: train, predict[/cyan]")
        sys.exit(1)


if __name__ == "__main__":
    main()
