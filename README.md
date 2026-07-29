# Natural Language Technologies Projects

A collection of Natural Language Processing projects developed during the
Natural Language Technologies course at the University of Turin.

The repository explores symbolic, statistical and neural approaches to NLP,
ranging from rule-based dialogue systems and lexical-semantic resources to
sentence embeddings, topic modeling, Large Language Models and multilingual
semantic analysis.

## 1. Lara Is Hiring — Rule-Based Dialogue System

`lara`

A character-based dialogue system in which Lara Croft interviews the user for
the role of archaeological assistant.

The application follows the traditional modular dialogue-system architecture:

```text
User Input
    ↓
Natural Language Understanding
    ↓
Dialogue Manager
    ↓
Natural Language Generation
    ↓
System Response
```

### Architecture

#### Natural Language Understanding

The NLU module uses spaCy to process free-form user input through:

- tokenization;
- lemmatization;
- dependency parsing;
- Named Entity Recognition;
- negation detection;
- boolean-intent recognition;
- keyword and answer extraction.

The system supports three question types:

- single-answer questions;
- boolean questions;
- multiple-answer questions.

It can identify correct, incorrect, ambiguous and partially complete answers.

#### Dialogue Manager

The Dialogue Manager is implemented as a finite-state machine with three main
states:

- `INTRO`
- `QUIZ`
- `END`

It manages:

- conversation state;
- user-name extraction;
- question progression;
- partial answers across multiple turns;
- ambiguity counters;
- incremental scoring;
- interaction history;
- performance analysis by topic;
- final verdict generation.

The system also derives Lara's current mood from the user's recent performance:

- positive after consecutive correct answers;
- negative after consecutive errors;
- neutral in mixed situations.

#### Natural Language Generation

The NLG component adopts a hybrid strategy:

- deterministic templates preserve Lara Croft's personality;
- SimpleNLG dynamically generates progress reports, scores and final feedback;
- template history prevents repetitive responses.

### Evaluation

The system was evaluated through three representative dialogue scenarios:

1. an ideal candidate;
2. a hesitant candidate;
3. an unsuccessful candidate.

The scenarios cover:

- correct and incorrect responses;
- boolean contradictions;
- incomplete multi-answer questions;
- noisy user input;
- repeated ambiguity;
- adaptive feedback;
- personalized final reports.

### Technologies

- Python
- spaCy
- SimpleNLG
- Regular expressions
- JSON
- Finite-state dialogue management

---

## 2. Lexical Semantics, Embeddings and LLMs

`di-caro`

This module contains five NLP laboratory experiments and a research-oriented
project on multilingual lexical ambiguity.

### Exercise 1 — WordNet and Distributional Embeddings

A Word Sense Disambiguation system combining:

- WordNet synsets, lemmas, glosses and examples;
- pretrained FastText embeddings;
- cosine similarity between context and candidate senses.

Each synset is represented by combining the embeddings of its lemmas, gloss and
usage examples.

The experiment highlights the advantages and limitations of combining symbolic
lexical resources with non-contextual distributional representations.

### Exercise 2 — Lexical and Semantic Similarity Between Definitions

An analysis of definitions belonging to concrete, abstract, general and
specific concepts.

Two complementary similarity measures are compared:

- lexical similarity using Jaccard overlap after stemming and stopword removal;
- semantic similarity using multilingual sentence embeddings and cosine
  similarity.

The results show that definitions can express highly similar meanings even when
they share relatively little vocabulary.

### Exercise 3 — Content-to-Form Semantic Retrieval

A reverse-dictionary system that attempts to retrieve the correct WordNet
concept starting from an Italian natural-language definition.

The pipeline combines:

- genus extraction through linguistic heuristics;
- candidate retrieval from WordNet and Open Multilingual WordNet;
- hyponym expansion;
- multilingual Sentence-BERT embeddings;
- semantic and lexical scoring.

The approach generally identifies the correct semantic domain, although exact
synset selection remains difficult for closely related concepts.

### Exercise 4 — Topic Modeling with BERTopic

An unsupervised topic-modeling experiment on a sample of 20,000 AG News
articles.

The pipeline includes:

1. document embeddings with `GTE-small`;
2. dimensionality reduction with UMAP;
3. density-based clustering with HDBSCAN;
4. topic extraction and interpretation with BERTopic.

The generated topics cover areas such as:

- sports;
- politics;
- business;
- technology;
- science and space.

The project also includes topic-keyword charts, document projections and
intertopic distance visualizations.

### Exercise 5 — LLM Prompting

Experiments with `Qwen/Qwen2.5-1.5B-Instruct` on two tasks:

- generating concise labels for BERTopic clusters;
- retrieving lexical items from natural-language definitions.

Two prompting strategies are compared:

- zero-shot prompting;
- few-shot prompting with semantic examples and explicit output constraints.

For the reverse-dictionary task, few-shot prompting improved accuracy from
36.60% to 50.33%.

The experiment demonstrates the importance of:

- prompt design;
- task-specific examples;
- constrained generation;
- output normalization and post-processing.

### Research Project — Multilingual Pseudowords

The research project investigates whether lexical ambiguity can be reduced by
combining words from different languages.

A multilingual pseudoword is constructed by pairing an Italian noun with a
French noun and associating it with the WordNet synsets shared by both words.

```text
pseudoword senses = Italian word senses ∩ French word senses
```

The pipeline uses WordNet and Open Multilingual WordNet to:

- index multilingual lexical inventories;
- identify candidate cross-lingual word pairs;
- compute shared synsets;
- filter noisy or excessively ambiguous candidates;
- evaluate ambiguity reduction.

Two evaluation metrics are considered:

- Ambiguity Reduction;
- Joint Reduction Score, introduced to reward balanced ambiguity reduction
  across both languages.

A total of 467 pseudowords were generated.

Average ambiguity decreased from:

- 9.24 synsets for Italian source words;
- 9.82 synsets for French words;

to:

- 3.15 shared synsets for the resulting pseudowords.

The results suggest that cross-lingual lexical variation can help construct
more stable and specific semantic units.

### Technologies

- Python
- NLTK
- WordNet
- Open Multilingual WordNet
- FastText
- Sentence Transformers
- Sentence-BERT
- BERTopic
- UMAP
- HDBSCAN
- Hugging Face Transformers
- Qwen2.5

---

## 3. WordNet, Word Sense Disambiguation and Language Models

`radicioni`

This module focuses on foundational symbolic and statistical NLP methods.

### Concept Similarity

Implementation and evaluation of three WordNet-based semantic similarity
measures:

- Wu & Palmer;
- Shortest Path;
- Leacock & Chodorow.

The measures are evaluated on WordSim353 by comparing automatically generated
scores with human judgments through:

- Pearson correlation;
- Spearman correlation.

The experiments highlight that taxonomic methods work well for `is-a`
relations, but struggle with functional and associative relationships not
directly represented in the WordNet hierarchy.

### Word Sense Disambiguation with Simplified Lesk

A knowledge-based Word Sense Disambiguation system implemented using
Simplified Lesk.

The system:

- normalizes and lemmatizes context words;
- removes stopwords;
- constructs sense signatures from WordNet glosses, examples and lemmas;
- selects the candidate synset with the highest lexical overlap;
- evaluates predictions against SemCor gold annotations.

Across ten randomized runs, the system achieved an average accuracy of
approximately 69.4%.

The experiment provides an interpretable baseline for understanding classical
knowledge-based WSD.

### Statistical Language Modeling

Bigram and trigram language models are trained on two different textual
domains:

#### Social Media

English tweets written by:

- Barack Obama;
- Cristiano Ronaldo.

The analysis compares stylistic properties such as:

- average tweet length;
- URLs;
- mentions;
- hashtags;
- retweets.

Generated tweets show that trigram models reproduce author-specific local
patterns more effectively than bigram models.

#### Literary Language

Bigram and trigram models are also trained on *Moby-Dick*.

The trigram model produces more locally coherent and stylistically plausible
passages, while still exhibiting the short-context limitations of classical
n-gram models.

Laplace smoothing is used to handle unseen n-grams.

### Technologies

- Python
- NLTK
- WordNet
- SemCor
- WordSim353
- pandas
- Statistical n-gram language models

---

## Academic Context

These projects were developed during the Natural Language Processing course
of the Master's Degree in Artificial Intelligence and Computer Systems at the
University of Turin.

The repository is intended to document practical experimentation with
different generations of NLP techniques: symbolic systems, statistical models,
distributional representations and Large Language Models.
