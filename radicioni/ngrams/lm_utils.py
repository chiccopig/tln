import re, random
from nltk.tokenize import word_tokenize
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

# ===================
# 1. PULIZIA TWEET
# ===================

URL_RE  = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
NUM_RE  = re.compile(r"\b\d+(\.\d+)?\b")

# normalizzazione del testo
def clean_tweet(t: str) -> str:
    t = t.strip() # rimozione spazi iniziali/finali
    t = URL_RE.sub("<URL>", t)
    t = USER_RE.sub("<USER>", t)
    t = NUM_RE.sub("<NUM>", t)
    return t

# tokenizzazione
def tokenize_tweet(t: str):
    return word_tokenize(t)

# ==============================================
# 2. ALLENAMENTO BIGRAM/TRIGRAM CON SMOOTHING
# ==============================================

def train_lm(texts, n=2):
    # preparazione dati
    tokenized = [tokenize_tweet(clean_tweet(t)) for t in texts if t.strip()]
    
    # crea n-gram e vocabolario, aggiungendo padding
    train_data, vocab = padded_everygram_pipeline(n, tokenized)
    # train_data: iteratore/generatore che fornisce gli everygrams al modello
    # vocab: raccoglie i token visti validi per il modello

    # creazione LM di ordine n con smoothing
    lm = Laplace(n)

    # allenamento del modello
    lm.fit(train_data, vocab)
    return lm

# ==============================
# 3. GENERAZIONE TESTO DAL LM
# ==============================

def generate_from_lm(lm, max_tokens=30, seed=0):
    random.seed(seed) # per riproducibilità
    context = ["<s>"] * (lm.order - 1) # <s> per bigram, <s> <s> per trigram
    out = []
    for _ in range(max_tokens):
        # calcola la distribuzione data la history P(parola|context)
        # estrae una parola in base a quella distribuzione
        nxt = lm.generate(1, text_seed=context)
        if nxt == "</s>": # se fine frase, si ferma
            break
        out.append(nxt) # aggiunta della parola generata all'output
        # aggiornamento del contesto lungo n-1 token (sliding window)
        if lm.order > 1:
            context = (context + [nxt])[-(lm.order - 1):]
    # ricostruzione stringa e sistemazione punteggiatura
    txt = " ".join(out).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return txt

# ==================================
# 4. CALCOLO STATISTICHE DI STILE
# ==================================

def style_stats(texts):
    n = len(texts) # n tweets
    toks = [tokenize_tweet(clean_tweet(t)) for t in texts]
    avg_len = sum(len(x) for x in toks) / n # lunghezza media (post processing)
    url_rate  = sum("<URL>" in clean_tweet(t) for t in texts) / n
    user_rate = sum("<USER>" in clean_tweet(t) for t in texts) / n
    hash_rate = sum("#" in t for t in texts) / n
    rt_rate   = sum(t.strip().startswith("RT") for t in texts) / n
    return {"N": n, "avg_tokens": avg_len, "URL%": url_rate, "@%": user_rate, "#%": hash_rate, "RT%": rt_rate}