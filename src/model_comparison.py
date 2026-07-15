import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

nlp = spacy.load("pl_core_news_sm")

def tokenize_pl(text):
    return [tok.text for tok in nlp(text)]

responses = pd.read_csv("responses.csv", sep=";", quotechar='"')
questions = pd.read_csv("questions.csv", sep=";", quotechar='"')
df = responses.merge(questions, on="question_id")

vectorizer = TfidfVectorizer().fit(df["question_text"])
Q = vectorizer.transform(df["question_text"])
R = vectorizer.transform(df["response_text"])
df["cosine_sim"] = [cosine_similarity(Q[i], R[i])[0, 0] for i in range(len(df))]

def jaccard(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / len(A | B) if (A | B) else 0.0
df["jaccard"] = df.apply(lambda r: jaccard(r["question_text"], r["response_text"]), axis=1)

df["resp_length"] = df["response_text"].str.split().apply(len)
df["unique_tokens"] = df["response_text"].str.split().apply(lambda toks: len(set(toks)))

smooth = SmoothingFunction().method1
df["bleu"] = df.apply(
    lambda r: sentence_bleu(
        [tokenize_pl(r["question_text"])],
        tokenize_pl(r["response_text"]),
        smoothing_function=smooth
    ),
    axis=1
)

if rouge_scorer:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    def comp_rouge(r):
        sc = scorer.score(r["question_text"], r["response_text"])
        return sc["rouge1"].fmeasure, sc["rouge2"].fmeasure, sc["rougeL"].fmeasure
    rouge_vals = df.apply(comp_rouge, axis=1, result_type="expand")
    df[["rouge1", "rouge2", "rougeL"]] = rouge_vals
else:
    df[["rouge1", "rouge2", "rougeL"]] = None

if bert_score:
    P, R, F = bert_score(
        df["response_text"].tolist(),
        df["question_text"].tolist(),
        lang="pl",
        rescale_with_baseline=False
    )
    df["bert_precision"] = P
    df["bert_recall"] = R
    df["bert_f1"] = F
else:
    df[["bert_precision", "bert_recall", "bert_f1"]] = None

columns_to_keep = [
    "question_id", "model",
    "cosine_sim", "jaccard", "resp_length", "unique_tokens",
    "bleu", "rouge1", "rouge2", "rougeL",
    "bert_precision", "bert_recall", "bert_f1"
]

df[columns_to_keep].to_csv("metrics_results.csv", index=False)

print(df.head(20))

metrics = [
    "cosine_sim", "jaccard", "resp_length", "unique_tokens",
    "bleu", "rouge1", "rouge2", "rougeL",
    "bert_precision", "bert_recall", "bert_f1"
]

for metric in metrics:
    pivot = df.pivot(index="question_id", columns="model", values=metric)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title(f"{metric}")
    plt.xlabel("Model")
    plt.ylabel("ID pytania")
    plt.tight_layout()
    plt.savefig(f"heatmap_{metric}.png")
    plt.close()