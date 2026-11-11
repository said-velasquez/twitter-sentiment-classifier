# Twitter Sentiment Classifier

A 4-class sentiment model for tweets (Positive, Negative, Neutral, Irrelevant) built with NLTK and scikit-learn, plus a small prototype that answers each tweet with a reply template that matches its predicted sentiment.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

## What it does

1. Cleans ~74,000 tweets with regular expressions: removes URLs, mentions, hashtags, symbols and English stop words.
2. Turns the text into 5,000 TF-IDF features.
3. Trains a Logistic Regression model on an 80/20 split.
4. Evaluates it with a classification report and a confusion matrix.
5. Uses the model in a reply function, tested on tweets from the validation set.

## Results

| Metric | Value |
|---|---|
| Accuracy (4 classes) | **68%** |
| F1 Negative | 0.74 |
| F1 Positive | 0.71 |
| F1 Neutral | 0.65 |
| F1 Irrelevant | 0.59 |

The confusion matrix shows that most errors come from the overlap between **Neutral** and **Irrelevant** tweets, which use very similar language.

## Project structure

```
twitter-sentiment-classifier/
├── notebooks/
│   └── twitter_sentiment_classifier.ipynb
├── data/
│   └── README.md        # how to get the dataset
├── requirements.txt
└── LICENSE
```

## Run it

```bash
git clone https://github.com/said-velasquez/twitter-sentiment-classifier.git
cd twitter-sentiment-classifier
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Download the dataset first and put the CSV files in `data/` (see [`data/README.md`](data/README.md)).

## Tech stack

Python, pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, Jupyter.

## Author

**Said Velásquez**, Computer Engineering student based in Madrid.
[LinkedIn](https://www.linkedin.com/in/said-velasquez) · [GitHub](https://github.com/said-velasquez)
