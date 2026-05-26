"""
Example workflow demonstrating the custom Naive Bayes classifier.

This script shows the complete pipeline from data loading to prediction
with explainability features.
"""

import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.naive_bayes import ExplainableNaiveBayes


def load_data(filepath):
    """Load PubMed RCT dataset."""
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


def main():
    print("=" * 70)
    print("  Explainable Naive Bayes - Complete Example")
    print("=" * 70)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # 1. Load Data
    print("\n[1/6] Loading data...")
    train_df = load_data(script_dir / 'train.txt')
    test_df = load_data(script_dir / 'test.txt')
    
    # Use a smaller subset for quick demo
    train_df = train_df.sample(n=10000, random_state=42)
    test_df = test_df.sample(n=2000, random_state=42)
    
    print(f"  [OK] Training samples: {len(train_df):,}")
    print(f"  [OK] Test samples: {len(test_df):,}")
    
    # 2. Feature Extraction
    print("\n[2/6] Extracting features...")
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents='unicode',
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_features=10000
    )
    
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_test = vectorizer.transform(test_df['sentence'])
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"  [OK] Vocabulary size: {X_train.shape[1]:,}")
    print(f"  [OK] Training matrix: {X_train.shape}")
    
    # 3. Train Model
    print("\n[3/6] Training Explainable Naive Bayes...")
    model = ExplainableNaiveBayes(alpha=1.0)
    model.fit(X_train, y_train, feature_names=vectorizer.get_feature_names_out())
    print(f"  [OK] Model trained on {len(model.classes_)} classes")
    
    # 4. Evaluate
    print("\n[4/6] Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  [OK] Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 5. Explainability: Top Features Per Class
    print("\n[5/6] Analyzing top features per class...")
    top_features = model.get_top_features_per_class(n=5)
    
    for class_name in sorted(top_features.keys()):
        words = [word for word, score in top_features[class_name]]
        print(f"  {class_name:12s}: {', '.join(words)}")
    
    # 6. Example Predictions with Explanations
    print("\n[6/6] Example predictions with explanations...")
    
    examples = [
        "A total of 125 patients were randomized to receive either treatment or placebo for 6 weeks.",
        "The results showed significant improvement with p < 0.001 compared to baseline.",
        "This study aimed to investigate the efficacy and safety of the new drug.",
        "Previous research has established the importance of early intervention.",
        "In conclusion, the treatment was effective and well-tolerated by patients."
    ]
    
    for i, text in enumerate(examples, 1):
        print(f"\n  Example {i}:")
        print(f"  Text: {text[:70]}...")
        
        explanation = model.explain_prediction(text, vectorizer, top_n=3)
        
        predicted = explanation['predicted_class']
        confidence = explanation['probabilities'][predicted]
        
        print(f"  Predicted: {predicted} (confidence: {confidence:.4f})")
        
        if explanation['top_words']:
            top_words = [word for word, _, _ in explanation['top_words'][:3]]
            print(f"  Key words: {', '.join(top_words)}")
    
    # 7. Pipeline Example
    print("\n" + "=" * 70)
    print("  Bonus: Scikit-Learn Pipeline Integration")
    print("=" * 70)
    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), min_df=3)),
        ('classifier', ExplainableNaiveBayes(alpha=1.0))
    ])
    
    print("\n  Training pipeline...")
    pipeline.fit(train_df['sentence'], train_df['label'])
    
    print("  Making predictions...")
    pipeline_pred = pipeline.predict(test_df['sentence'])
    pipeline_acc = accuracy_score(test_df['label'], pipeline_pred)
    
    print("  [OK] Pipeline accuracy: {pipeline_acc:.4f}")
    print("\n  Pipeline works with scikit-learn!")
    
    print("\n" + "=" * 70)
    print("  Example Complete")
    print("=" * 70)
    print("\n  Next steps:")
    print("  - Try: python cli.py train")
    print("  - Try: python cli.py predict \"Your text here\"")
    print("  - Open: jupyter notebook demo.ipynb")
    print()


if __name__ == "__main__":
    main()
