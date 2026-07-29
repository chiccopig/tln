import math
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from scipy.stats import pearsonr, spearmanr

# ===================
# 1. SETUP WORDNET
# ===================

try:
    wn.synsets("dog")
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# =============================
# 2. CALCOLO DEPTHMAX (NOMI)
# =============================

# calcolo della profondità massima della tassonomia nominale
DEPTH_MAX = max(s.min_depth() for s in wn.all_synsets(pos='n'))

# ==========================
# 3. MISURE DI SIMILARITÀ
# ==========================

def wu_palmer_score(s1, s2):
    """
    Wu & Palmer:
    sim = 2 * depth(LCS) / (depth(s1) + depth(s2))
    """
    # recupera gli iperonimi comuni più bassi (più specifici)
    lcs_list = s1.lowest_common_hypernyms(s2)
    if not lcs_list:
        return 0

    # se ci sono più LCS, si sceglie quello più profondo
    lcs = max(lcs_list, key=lambda s: s.min_depth())

    # calcolo profondità (aggiungendo 1 per evitare lo 0 della root)
    depth_lcs = lcs.min_depth() + 1
    depth_s1 = s1.min_depth() + 1
    depth_s2 = s2.min_depth() + 1

    return (2 * depth_lcs) / (depth_s1 + depth_s2)

def shortest_path_score(s1, s2):
    """
    Shortest Path:
    sim = 2 * depthMax - len(s1, s2)
    dove len è il numero di archi
    """
    # distanza minima (n archi) tra i due synset nella gerarchia
    dist = s1.shortest_path_distance(s2)
    if dist is None:
        return 0

    # più la distanza è piccola, più la similarità è alta
    return (2 * DEPTH_MAX) - dist

def leacock_chodorow_score(s1, s2):
    """
    Leacock & Chodorow:
    sim = -log( len(s1, s2) / (2 * depthMax) )
    """
    # distanza minima tra i synset
    dist = s1.shortest_path_distance(s2)
    if dist is None:
        return 0

    # +1 per evitare log(0) quando dist == 0 (synset identici)
    return -math.log((dist + 1) / (2 * DEPTH_MAX))

# ===========================
# 4. SIMILARITÀ TRA PAROLE
# ===========================

def word_similarity(word1, word2, sim_func):
    """
    Calcola la similarità tra due termini (word1, word2) come:
    massimo tra tutte le coppie di sensi nominali dei due termini
    """
    # recupero dei synset nominali dei due termini
    synsets1 = wn.synsets(word1, pos='n')
    synsets2 = wn.synsets(word2, pos='n')

    max_score = 0
    # doppio ciclo per trovare la combinazione di sensi più simile
    for s1 in synsets1:
        for s2 in synsets2:
            score = sim_func(s1, s2)
            if score > max_score:
                max_score = score

    return max_score

# ========================
# 5. DATASET WordSim353
# ========================

df = pd.read_csv("WordSim353.csv")

wup_scores = []
path_scores = []
lch_scores = []

print("Calcolo similarità...")

# iterazione sulle 353 coppie
for _, row in df.iterrows():
    w1 = row["Word 1"]
    w2 = row["Word 2"]

    # calcolo della similarità tra parole usando ciascuna misura
    wup_scores.append(word_similarity(w1, w2, wu_palmer_score))
    path_scores.append(word_similarity(w1, w2, shortest_path_score))
    lch_scores.append(word_similarity(w1, w2, leacock_chodorow_score))

df["Wu_Palmer"] = wup_scores
df["Shortest_Path"] = path_scores
df["Leacock_Chodorow"] = lch_scores

# =================
# 6. VALUTAZIONE
# =================

human = df["Human (mean)"]

print("\n--- CORRELAZIONI ---")

print("Wu & Palmer")
print(" Pearson :", pearsonr(df["Wu_Palmer"], human)[0])
print(" Spearman:", spearmanr(df["Wu_Palmer"], human)[0])

print("\nShortest Path")
print(" Pearson :", pearsonr(df["Shortest_Path"], human)[0])
print(" Spearman:", spearmanr(df["Shortest_Path"], human)[0])

print("\nLeacock & Chodorow")
print(" Pearson :", pearsonr(df["Leacock_Chodorow"], human)[0])
print(" Spearman:", spearmanr(df["Leacock_Chodorow"], human)[0])

print("\nPrime righe:")
print(df[["Word 1", "Word 2", "Human (mean)",
          "Wu_Palmer", "Shortest_Path", "Leacock_Chodorow"]].head(10))