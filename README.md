# 🤖 AI vs Human Text Detector (Feature Engineering)

A machine learning project that distinguishes between human-written and AI-generated text by analyzing linguistic features.

## 🚀 Features
- **Perplexity Score:** Measures how "surprised" a model is by the text (AI text usually has lower perplexity).
- **Burstiness:** Analyzes the variation in sentence structure.
- **Readability Metrics:** Uses Flesch-Kincaid scores to determine complexity.
- **N-gram Analysis:** Detects repetitive patterns common in AI text.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** NLTK, Scikit-learn, Pandas, NumPy, Matplotlib
- **Visualization:** Matplotlib/Seaborn for feature distribution graphs.

## 📊 How It Works
1.  Input a text sample.
2.  The system extracts 15+ linguistic features.
3.  A Random Forest Classifier (or your specific model) predicts the probability of it being AI-generated.

## 📈 Results
*(You can add a screenshot of your graphs here later!)*