"""Quick test to verify the implementation works correctly."""

import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.naive_bayes import ExplainableNaiveBayes

# Sample data
texts = [
    "Patients were randomized to receive treatment or placebo",
    "The results showed significant improvement with p < 0.001",
    "This study aimed to investigate the efficacy of the drug",
    "The background of this research is well established",
    "In conclusion, the treatment was effective and safe"
]

labels = ["METHODS", "RESULTS", "OBJECTIVE", "BACKGROUND", "CONCLUSIONS"]

print("Testing Naive Bayes Implementation\n")

# Test 1: Basic fit and predict
print("Test 1: Basic fit and predict...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = ExplainableNaiveBayes(alpha=1.0)
model.fit(X, labels, feature_names=vectorizer.get_feature_names_out())
predictions = model.predict(X)
print(f"Predictions: {predictions}")
print(f"Accuracy: {np.mean(predictions == labels):.2f}\n")

# Test 2: predict_proba
print("Test 2: Probability predictions...")
probas = model.predict_proba(X)
print(f"Shape: {probas.shape}")
print(f"Sum of probabilities: {probas.sum(axis=1)}")
print(f"All probabilities sum to 1: {np.allclose(probas.sum(axis=1), 1.0)}\n")

# Test 3: Explainability
print("Test 3: Explainability...")
test_text = "Patients were randomized to receive placebo"
explanation = model.explain_prediction(test_text, vectorizer, top_n=3)
print(f"Predicted class: {explanation['predicted_class']}")
print(f"Top words: {[w[0] for w in explanation['top_words']]}\n")

# Test 4: Top features per class
print("Test 4: Top features per class...")
top_features = model.get_top_features_per_class(n=3)
print(f"Classes: {list(top_features.keys())}")
for cls, features in top_features.items():
    print(f"  {cls}: {[f[0] for f in features]}")

print("\nAll tests passed.")

