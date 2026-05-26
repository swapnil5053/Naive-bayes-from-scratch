"""
Scikit-Learn compatible Multinomial Naive Bayes classifier with explainability features.

This module provides a vectorized implementation of Multinomial Naive Bayes
that works with scikit-learn pipelines.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse import csr_matrix


class ExplainableNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Multinomial Naive Bayes classifier with explainability features.
    
    Implements the Multinomial Naive Bayes algorithm with:
    - Scikit-learn API compatibility (fit, predict, predict_proba)
    - Vectorized NumPy operations for efficient predictions
    - Explainability methods for interpretable predictions
    - Laplace smoothing for handling unseen features
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
        Setting alpha=0 means no smoothing.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class (smoothed).
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Empirical log probability of features given a class, P(x_i|y).
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_ : ndarray of shape (n_features,), optional
        Feature names for explainability.
    vocabulary_ : dict, optional
        Mapping from feature names to indices.
    
    Examples
    --------
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> 
    >>> # Create a pipeline
    >>> pipeline = Pipeline([
    ...     ('vectorizer', CountVectorizer()),
    ...     ('classifier', ExplainableNaiveBayes(alpha=1.0))
    ... ])
    >>> 
    >>> # Fit and predict
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def fit(self, X, y, feature_names: Optional[np.ndarray] = None):
        """
        Fit Multinomial Naive Bayes classifier.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors (typically from CountVectorizer or TfidfVectorizer).
        y : array-like of shape (n_samples,)
            Target values.
        feature_names : array-like of shape (n_features,), optional
            Feature names for explainability. If using CountVectorizer,
            pass vectorizer.get_feature_names_out().
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert to numpy array if needed
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        
        # Store classes and feature count
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        # Store feature names for explainability
        if feature_names is not None:
            self.feature_names_ = np.array(feature_names)
            self.vocabulary_ = {name: idx for idx, name in enumerate(feature_names)}
        else:
            self.feature_names_ = None
            self.vocabulary_ = None
        
        # Initialize arrays for log probabilities
        n_classes = len(self.classes_)
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_in_))
        
        # Compute class priors and feature likelihoods (fully vectorized)
        for idx, c in enumerate(self.classes_):
            # Get all samples belonging to class c
            class_mask = (y == c)
            X_c = X[class_mask]
            
            # Compute log prior: log(P(c))
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])
            
            # Compute feature counts for this class
            if hasattr(X_c, 'toarray'):  # Sparse matrix
                feature_count = np.asarray(X_c.sum(axis=0)).ravel()
            else:  # Dense array
                feature_count = X_c.sum(axis=0)
            
            # Total word count in this class
            total_count = feature_count.sum()
            
            # Apply Laplace smoothing and compute log likelihood
            # log(P(w_i | c)) = log((count(w_i, c) + alpha) / (total_count + alpha * n_features))
            smoothed_count = feature_count + self.alpha
            smoothed_total = total_count + self.alpha * self.n_features_in_
            
            self.feature_log_prob_[idx, :] = np.log(smoothed_count / smoothed_total)
        
        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test vectors.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        # Get log probabilities for all classes
        log_probs = self._compute_log_probs(X)
        
        # Return class with highest log probability (fully vectorized)
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Return probability estimates for the test vector X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test vectors.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        # Get log probabilities
        log_probs = self._compute_log_probs(X)
        
        # Convert to probabilities using log-sum-exp trick for numerical stability
        # P(c|x) = exp(log P(c|x)) / sum_c' exp(log P(c'|x))
        log_probs_max = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - log_probs_max)
        proba = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)
        
        return proba
    
    def _compute_log_probs(self, X) -> np.ndarray:
        """
        Compute log probabilities for all classes (fully vectorized).
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input vectors.
        
        Returns
        -------
        log_probs : ndarray of shape (n_samples, n_classes)
            Log probabilities for each class.
        """
        # Vectorized computation: log P(c|x) = log P(c) + sum_i count(w_i) * log P(w_i|c)
        # This is a matrix multiplication: X @ feature_log_prob_.T
        if hasattr(X, 'toarray'):  # Sparse matrix
            log_likelihood = X @ self.feature_log_prob_.T
        else:  # Dense array
            log_likelihood = X @ self.feature_log_prob_.T
        
        # Add class priors
        log_probs = log_likelihood + self.class_log_prior_
        
        return log_probs
    
    def explain_prediction(
        self, 
        text: str, 
        vectorizer, 
        top_n: int = 5
    ) -> Dict[str, any]:
        """
        Explain which words contributed most to the prediction.
        
        This method provides interpretability by showing which features
        (words) had the highest impact on the classification decision.
        
        Parameters
        ----------
        text : str
            The input text to explain.
        vectorizer : CountVectorizer or TfidfVectorizer
            The fitted vectorizer used to transform the text.
        top_n : int, default=5
            Number of top contributing words to return.
        
        Returns
        -------
        explanation : dict
            Dictionary containing:
            - 'predicted_class': The predicted class label
            - 'probabilities': Probability distribution across all classes
            - 'top_words': List of (word, contribution_score) tuples
            - 'class_scores': Log probability scores for each class
        """
        # Transform text
        X = vectorizer.transform([text])
        
        # Get prediction and probabilities
        predicted_class = self.predict(X)[0]
        probabilities = self.predict_proba(X)[0]
        
        # Get log probabilities for each class
        log_probs = self._compute_log_probs(X)[0]
        
        # Find the predicted class index
        predicted_idx = np.where(self.classes_ == predicted_class)[0][0]
        
        # Get feature contributions for the predicted class
        # Contribution = count(word) * log P(word|class)
        if hasattr(X, 'toarray'):
            word_counts = X.toarray()[0]
        else:
            word_counts = X[0]
        
        contributions = word_counts * self.feature_log_prob_[predicted_idx, :]
        
        # Get top contributing features
        top_indices = np.argsort(contributions)[::-1][:top_n]
        
        # Get feature names
        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = vectorizer.get_feature_names_out()
        
        # Build explanation
        top_words = [
            (feature_names[idx], contributions[idx], word_counts[idx])
            for idx in top_indices
            if word_counts[idx] > 0  # Only include words that appear in the text
        ]
        
        return {
            'predicted_class': predicted_class,
            'probabilities': {
                class_name: prob 
                for class_name, prob in zip(self.classes_, probabilities)
            },
            'top_words': top_words,
            'class_scores': {
                class_name: score 
                for class_name, score in zip(self.classes_, log_probs)
            }
        }
    
    def get_top_features_per_class(self, n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get the most indicative features for each class.
        
        This method returns the top n features (words) that are most
        strongly associated with each class based on log-likelihood ratios.
        
        Parameters
        ----------
        n : int, default=10
            Number of top features to return per class.
        
        Returns
        -------
        top_features : dict
            Dictionary mapping class names to lists of (feature, score) tuples.
        """
        if self.feature_names_ is None:
            raise ValueError(
                "Feature names not available. Pass feature_names during fit() "
                "or use a vectorizer with get_feature_names_out()."
            )
        
        top_features = {}
        
        for idx, class_name in enumerate(self.classes_):
            # Get log probabilities for this class
            class_log_probs = self.feature_log_prob_[idx, :]
            
            # Get top n features
            top_indices = np.argsort(class_log_probs)[::-1][:n]
            
            top_features[class_name] = [
                (self.feature_names_[i], class_log_probs[i])
                for i in top_indices
            ]
        
        return top_features
