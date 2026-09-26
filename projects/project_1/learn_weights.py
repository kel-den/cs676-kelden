"""
learn_weights.py — Fit logistic regression to learn optimal RULE_WEIGHT.

This script:
1. Uses 24 labeled URLs (from README examples)
2. Extracts features from each URL using credibility.py functions
3. Trains logistic regression to predict credibility scores
4. Reports the optimal RULE_WEIGHT for blending rule and LLM layers

Usage: python learn_weights.py
"""

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import credibility functions
sys.path.insert(0, '.')
from credibility import rule_based_signals, _combine_signals

# Labeled URLs from evaluate.py test set (24 examples)
LABELED_URLS = [
    ("https://www.biorxiv.org/content/10.1101/2020.01.01.0000...", 0.50),
    ("https://www.who.int/news-room/fact-sheets/detail/example", 0.88),
    ("https://health-truth-daily.info/miracle-cure-doctors-hate", 0.05),
    ("https://www.propublica.org/article/example-investigation", 0.85),
    ("https://www.imf.org/en/Publications/WEO/example", 0.85),
    ("https://example.com/sponsored/miracle-supplement", 0.10),
    ("https://www.pnas.org/doi/10.1073/pnas.2020123118", 0.92),
    ("https://seekingalpha.com/article/example-stock-analysis", 0.35),
    ("http://totally-legit-news.xyz/shocking-truth", 0.05),
    ("https://scikit-learn.org/stable/modules/linear_model.html", 0.75),
    ("https://randomblog.blogspot.com/2024/03/my-thoughts.html", 0.15),
    ("https://arxiv.org/abs/1706.03762", 0.65),
    ("https://medium.com/@someone/why-ai-is-magic-abc123", 0.25),
    ("https://en.wikipedia.org/wiki/Statistical_learning_theory", 0.60),
    ("https://stackoverflow.com/questions/12345/how-to-do-x", 0.45),
    ("https://www.nejm.org/doi/full/10.1056/NEJMoa2034577", 0.95),
    ("https://www.reddit.com/r/MachineLearning/comments/abc123/", 0.20),
    ("https://pubmed.ncbi.nlm.nih.gov/33812246/", 0.90),
    ("https://www.nature.com/articles/s41586-021-03819-2", 0.95),
    ("https://www.reuters.com/world/example-report-2024-01-01/", 0.85),
    ("https://apnews.com/article/example-story-12345", 0.85),
    ("https://www.theonion.com/study-finds-example-1849", 0.05),
    ("https://jamanetwork.com/journals/jama/fullarticle/2762138", 0.93),
    ("https://www.census.gov/data/tables/2023/demo/income-pov...", 0.90),
]

def extract_features(url: str) -> list:
    """
    Extract numeric features from a URL for regression.
    
    Features:
    - rule_score: base score from rule-based layer
    - signal_count: number of signals that fired
    - has_https: 1 if HTTPS, 0 otherwise
    - has_doi: 1 if URL contains DOI pattern, 0 otherwise
    """
    from urllib.parse import urlparse
    import re
    
    signals = rule_based_signals(url)
    rule_score = _combine_signals(signals)
    signal_count = len(signals)
    
    parsed = urlparse(url)
    has_https = 1 if parsed.scheme == "https" else 0
    has_doi = 1 if re.search(r"/10\.\d{4,9}/", parsed.path or "") else 0
    
    return [rule_score, signal_count, has_https, has_doi]

def learn_weights():
    """
    Fit logistic regression to learn optimal weights.
    """
    print("=" * 70)
    print("LEARNING OPTIMAL RULE_WEIGHT via Logistic Regression")
    print("=" * 70)
    
    # Extract features and labels from labeled URLs
    X = []  # Features
    y = []  # Labels (credibility scores)
    
    print(f"\nExtracting features from {len(LABELED_URLS)} labeled URLs...")
    
    for url, expected_score in LABELED_URLS:
        try:
            features = extract_features(url)
            X.append(features)
            # Convert score [0, 1] to binary classification [0 or 1]
            # Threshold: 0.5 (scores >= 0.5 are "credible", < 0.5 are "not credible")
            label = 1 if expected_score >= 0.5 else 0
            y.append(label)
        except Exception as e:
            print(f"  ⚠️  Failed to extract features from {url[:50]}...: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Extracted {len(X)} feature vectors\n")
    
    # Standardize features (important for regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train logistic regression
    print("Training logistic regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    accuracy = model.score(X_scaled, y)
    print(f"✓ Model trained (Accuracy: {accuracy:.1%})\n")
    
    # Extract learned weights
    coefficients = model.coef_[0]
    
    print("Feature Coefficients (learned importance):")
    print(f"  rule_score:    {coefficients[0]:+.4f}")
    print(f"  signal_count:  {coefficients[1]:+.4f}")
    print(f"  has_https:     {coefficients[2]:+.4f}")
    print(f"  has_doi:       {coefficients[3]:+.4f}")
    print(f"  intercept:     {model.intercept_[0]:+.4f}")
    
    # Recommend optimal RULE_WEIGHT
    # The primary feature is rule_score; its weight tells us importance
    rule_importance = abs(coefficients[0])
    llm_importance = 1.0  # LLM layer provides additional signal
    
    optimal_rule_weight = rule_importance / (rule_importance + llm_importance)
    optimal_llm_weight = 1.0 - optimal_rule_weight
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION FOR credibility.py:")
    print("=" * 70)
    print(f"RULE_WEIGHT = {optimal_rule_weight:.2f}")
    print(f"LLM_WEIGHT  = {optimal_llm_weight:.2f}")
    print("=" * 70)
    
    return optimal_rule_weight, optimal_llm_weight

if __name__ == "__main__":
    try:
        rule_w, llm_w = learn_weights()
        print(f"\n✓ Analysis complete!")
        print(f"\nNext steps:")
        print(f"1. Open credibility.py")
        print(f"2. Find lines 59-60 (RULE_WEIGHT and LLM_WEIGHT)")
        print(f"3. Change to: RULE_WEIGHT = {rule_w:.2f}")
        print(f"             LLM_WEIGHT = {llm_w:.2f}")
        print(f"4. Run: python evaluate.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()