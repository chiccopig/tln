import random, re
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tree import Tree
from statistics import mean, pstdev

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("semcor")

from nltk.corpus import semcor

# ====================
# 1. PRE-PROCESSING
# ====================

# costanti globali: stopwords e lemmatizzatore
STOP = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()

# tokenizzazione stringa (solo sequenze alfabetiche, minuscole)
def tokenize(text: str):
    return re.findall(r"[a-z]+", text.lower()) # token puliti

# rimozione stopwords e lemmatizzazione
def normalize_words(words):
    out = []
    for w in words:
        if w in STOP:
            continue
        out.append(LEMM.lemmatize(w))
    return out

# =======================
# 2. ALGORITMO DI LESK
# =======================

def simplified_lesk(word, sentence_tokens, pos="n"):
    """
    Cerca di indovinare il senso di 'word' confrontando il suo contesto (frase)
    con le definizioni (gloss) di WordNet
    """
    # normalizzazione per gestire parole composte
    word = word.lower().replace(" ", "_")
    # costruzione del contesto
    context = set(normalize_words([t.lower() for t in sentence_tokens]))
    # recupero synset della parola target da WordNet
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return None, 0

    # default: primo senso (spesso è il most frequent sense)
    best = synsets[0]
    max_ov = -1 # in modo che il primo senso venga considerato

    # per ciascun senso, costruzione signature e calcolo overlap
    for s in synsets:
        # costruzione signature del senso corrente
        signature = []
        signature += tokenize(s.definition()) # parole della definizione (gloss)
        for ex in s.examples():               # parole degli esempi
            signature += tokenize(ex)
        # aggiunta dei sinonimi associato al senso
        signature += [x.replace("_", " ") for x in s.lemma_names()]
        
        # normalizzazione signature
        signature = set(normalize_words(signature))
        
        # calcolo overlap (intersesione signature-contesto)
        overlap = len(signature & context)

        # scelta del senso con overlap massimo
        if overlap > max_ov:
            max_ov = overlap
            best = s

    return best, max_ov

# =============================================
# 3. ESTRAZIONE TOKENS + CANDIDATI DA SEMCOR
# =============================================

# estrazione tokens + candidati da SemCor
def flatten_sentence(sent):
    """
    Trasforma una frase di SemCor (struttura annidata) in:
    - tokens: lista piatta di stringhe (frase lineare)
    - candidates: lista di tuple (surface, gold_syn)
    """
    tokens = []     # conterrà tutte le parole della frase per il contesto
    candidates = [] # conterrà tuple (parola, senso corretto) per i test

    for chunk in sent:
        # se il chunk è un Tree, allora contiene un'annotazione semantica
        if isinstance(chunk, Tree):
            # estrazione delle parole (foglie dell'albero)
            leaves = chunk.leaves()
            words = [w for w in leaves if isinstance(w, str)]
            tokens.extend(words)

            # etichetta semantica (gold sense)
            gold = chunk.label()
            # se l'etichetta è un lemma WordNet, si può ottenere il synset
            if hasattr(gold, "synset"):
                gold_syn = gold.synset() # etichetta vera
                # recupero del POS tag del chunk (se tag='both', il figlio Tree(pos, ww))
                pos_tag = None
                if len(chunk) > 0 and isinstance(chunk[0], Tree):
                    pos_tag = chunk[0].label()
                
                # unione delle parole per gestire i multi-word
                surface = "_".join(words).lower()
                
                # se sostantivo, si aggiunge ai candidati
                if pos_tag and str(pos_tag).startswith("NN"):
                    candidates.append((surface, gold_syn))
        
        # se il chunk non è annotato semanticamente
        else:
            if isinstance(chunk, list):
                tokens.extend([w for w in chunk if isinstance(w, str)])
            elif isinstance(chunk, str):
                tokens.append(chunk)
    
    return tokens, candidates

# ==========================
# 4. RUN DELL'ESPERIMENTO
# ==========================

def one_run(n_sent=50, seed=None, verbose=False, show_n=10):
    rng = random.Random(seed)

    # scelta casuale di 50 frasi di SemCor
    sents = semcor.tagged_sents(tag="both")
    chosen = rng.sample(list(sents), n_sent)

    correct = 0 # predizioni corrette
    total = 0   # predizioni totali
    shown = 0   # esempi mostrati in debug

    # per ciascuna frase: si estrae un candidato e si sceglie 1 sostantivo random
    for sent in chosen:
        tokens, candidates = flatten_sentence(sent)
        # se non ci sono sostantivi annotati, salta
        if not candidates:
            continue
        
        # scelta random di un sostantivo target
        target_word, gold_syn = rng.choice(candidates)

        # predizione del senso con Lesk
        pred_syn, overlap = simplified_lesk(target_word, tokens, pos="n")

        total += 1 # numero di predizioni tentate
        
        if pred_syn is None:
            continue
        else:
            # verifica se il senso predetto coincide con quello annotato
            ok = (pred_syn == gold_syn)

        if ok:
            correct += 1 # numero di predizioni corrette

        # debug
        if verbose and shown < show_n:
            shown += 1
            print("-" * 70)
            print("Sentence:", " ".join(tokens))
            print("Target  :", target_word)
            print("Overlap :", overlap)
            print("Gold    :", gold_syn.name(), "—", gold_syn.definition())
            if pred_syn is None:
                print("Pred    :", None)
            else:
                print("Pred    :", pred_syn.name(), "—", pred_syn.definition())
            print("Correct :", ok)

    # accuracy: predizioni corrette / predizioni totali
    return (correct / total) if total > 0 else 0.0

# ==========================
# 5. VALUTAZIONE
# ==========================

# prima run con output di debug
print("\nAvvio prima run ...")
accuracy1 = one_run(n_sent=50, seed=0, verbose=True, show_n=8)

# 10 run indipendenti per calcolare media e dev std
print("-" * 70)
print("\nCalcolo accuracy per 10 run...")

scores = []
for i in range(10):
    acc = one_run(n_sent=50, seed=i)
    scores.append(acc)
    print(f" Run {i+1}/10: Accuratezza = {acc:.2%}")

# calcolo statistiche finali
print("-" * 70)
print(f"Media accuracy:      {mean(scores):.2%}")
print(f"Deviazione standard: {pstdev(scores):.4f}")
print(f"Range:               [{min(scores):.2%} - {max(scores):.2%}]")