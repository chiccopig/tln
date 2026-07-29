import pandas as pd
import nltk
from lm_utils import train_lm, generate_from_lm, style_stats

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# =====================================
# 1. CARICAMENTO TWEET ED ESTRAZIONE
# =====================================

df = pd.read_csv("tweets.csv")

df_obama = df[(df["author"] == "BarackObama") & (df["language"] == "en")]
df_cristiano  = df[(df["author"] == "Cristiano") & (df["language"] == "en")]

# estrazione tweet (lista di stringhe)
tweets_obama = df_obama["content"].dropna().astype(str).tolist()
tweets_cristiano = df_cristiano["content"].dropna().astype(str).tolist()

# =========================
# 2. ALLENAMENTO MODELLI
# =========================

lmA_2 = train_lm(tweets_obama, n=2)
lmA_3 = train_lm(tweets_obama, n=3)
lmB_2 = train_lm(tweets_cristiano, n=2)
lmB_3 = train_lm(tweets_cristiano, n=3)

# ================================
# 3. STAMPA TWEET E STATISTICHE
# ================================

def print_samples(title, lm):
    print("\n" + "="*10, title, "="*10)
    for i in range(5):
        print("-", generate_from_lm(lm, max_tokens=25, seed=i))

print("="*10, "Tweet stats", "="*10)
print("Obama stats:    ", style_stats(tweets_obama))
print("Cristiano stats:", style_stats(tweets_cristiano))

print_samples("Obama / BIGRAM", lmA_2)
print_samples("Obama / TRIGRAM", lmA_3)
print_samples("Cristiano / BIGRAM", lmB_2)
print_samples("Cristiano / TRIGRAM", lmB_3)