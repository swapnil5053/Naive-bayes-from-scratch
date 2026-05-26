# Custom Naive Bayes for Medical Text Classification

A from-scratch implementation of Multinomial Naive Bayes for classifying sentences in medical research abstracts.

## Overview

This lab program classifies sentences from medical research abstracts into their structural roles (Background, Methods, Results, etc.). Originally a machine learning lab assignment, it has been extended with scikit-learn compatibility, explainability features, and a command-line interface.

## Dataset

**PubMed 20k RCT** - Sentences from randomized controlled trial abstracts in PubMed (a biomedical literature database).

- **Training set**: 180,040 sentences
- **Development set**: 30,212 sentences
- **Test set**: 30,135 sentences

### Classes (5 categories):
- BACKGROUND - Context and previous research
- OBJECTIVE - Goals of the study
- METHODS - How the study was conducted
- RESULTS - What was found
- CONCLUSIONS - What it means

## Implementation

### Features:
- Custom Naive Bayes classifier built from scratch
- Scikit-learn API compatibility (BaseEstimator, ClassifierMixin)
- Count Vectorization with n-grams (1-2 grams)
- Laplace smoothing for handling unseen words
- Log-probability calculations to prevent underflow
- Vectorized NumPy operations for efficiency
- Explainability methods to see which words influenced predictions

### Algorithm:

**Bayes' Theorem:**
```
P(class | document) ∝ P(class) × ∏ P(word | class)^count(word)
```

**Log-Space Conversion:**
```
log P(class | document) = log P(class) + Σ count(word) × log P(word | class)
```

**Laplace Smoothing:**
```
P(word | class) = (count(word, class) + α) / (total_words_in_class + α × vocab_size)
```

## Installation

```bash
git clone https://github.com/swapnil5053/Naive-Bayes-Text-Classification-lab.git
cd Naive-Bayes-Text-Classification-lab-main
pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0 scipy>=1.7.0 rich>=10.0.0
```

For Jupyter notebook support:
```bash
pip install jupyter>=1.0.0 matplotlib>=3.4.0 seaborn>=0.11.0
```

## Usage

### Command-Line Interface

**Train the model:**
```bash
python cli.py train
```

**Classify a sentence:**
```bash
python cli.py predict "A total of 125 patients were randomized to receive treatment or placebo."
```

### Python API

```python
from sklearn.feature_extraction.text import CountVectorizer
from src.naive_bayes import ExplainableNaiveBayes

# Prepare data
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)

# Train model
model = ExplainableNaiveBayes(alpha=1.0)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Files

- `src/naive_bayes.py` - Core classifier implementation
- `cli.py` - Command-line interface
- `example.py` - Complete workflow example
- `demo.ipynb` - Interactive Jupyter notebook
- `Naive_Bayes_Classifier_Boilerplate.ipynb` - Original lab notebook
- `train.txt` - Training data
- `dev.txt` - Development data
- `test.txt` - Test data

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 74.4% |
| Macro F1 | 67.8% |

**Per-class results:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| BACKGROUND | 0.53 | 0.57 | 0.55 |
| CONCLUSIONS | 0.61 | 0.71 | 0.65 |
| METHODS | 0.84 | 0.84 | 0.84 |
| OBJECTIVE | 0.52 | 0.53 | 0.52 |
| RESULTS | 0.88 | 0.78 | 0.83 |

