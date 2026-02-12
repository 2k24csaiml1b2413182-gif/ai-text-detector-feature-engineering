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
<img width="1000" height="600" alt="graph_avg_sent_len" src="https://github.com/user-attachments/assets/d070d4f8-1478-4a34-bb9b-3d5d49492ab2" />
<img width="1000" height="600" alt="graph_burstiness" src="https://github.com/user-attachments/assets/6fdd173c-cee7-4ecd-8688-b5854c644461" />
<img width="1000" height="600" alt="graph_readability" src="https://github.com/user-attachments/assets/304c86d8-2b54-4196-8fe5-f259b031b5d1" />
<img width="1000" height="600" alt="graph_vocab_richness" src="https://github.com/user-attachments/assets/61aa3011-999d-4175-8ea8-d87e99b22d42" />
