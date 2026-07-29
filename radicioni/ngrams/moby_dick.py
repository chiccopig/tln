import nltk
from nltk.tokenize import sent_tokenize
from lm_utils import train_lm, generate_from_lm

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ==========================================
# 1. CARICAMENTO FILE E CREAZIONE DATASET
# ==========================================

with open("moby-dick.txt", "r", encoding="utf-8") as f:
    text = f.read()

# spezza il testo in frasi
sentences = sent_tokenize(text)

# filtra frasi troppo corte
sentences = [s for s in sentences if len(s.split()) >= 5]

print("Numero frasi:", len(sentences))
print("Esempio frase:", sentences[99])

# =========================
# 2. ALLENAMENTO MODELLI
# =========================

lm2 = train_lm(sentences, n=2) # bigram
lm3 = train_lm(sentences, n=3) # trigram

# =======================
# 3. GENERAZIONE TESTI
# =======================

def print_samples(title, lm, k=5, max_tokens=60):
    print("\n" + "="*10, title, "="*10)
    for i in range(k):
        print("-", generate_from_lm(lm, max_tokens=max_tokens, seed=i))

print_samples("Moby-Dick / BIGRAM", lm2, k=5, max_tokens=60)
print_samples("Moby-Dick / TRIGRAM", lm3, k=5, max_tokens=60)