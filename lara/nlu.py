import re
import sys
import spacy

class NLUEngine:
    """
    Analizza le risposte dell'utente, determinandone correttezza e ambiguità
    """

    def __init__(self):
        try:
            # caricamento del modello inglese
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("\n[CRITICAL ERROR] Language model not found.")
            print("Please run: python -m spacy download en_core_web_sm")
            sys.exit(1)

    def analyze_answer(self, user_text, question):
        # SpaCy processa il testo per ottenere lemmi, dipendenze e tag
        doc = self.nlp(user_text.lower())
        question_type = question["type"]

        if question_type == "single":
            return self._analyze_single(doc, user_text, question)
        elif question_type == "boolean":
            return self._analyze_boolean(doc, question)
        elif question_type == "multiple":
            return self._analyze_multiple(user_text, question)
        return False
    
    def _analyze_single(self, doc, original_text, question):
        """
        Analizza risposte a domande a risposta singola.
        Verifica la presenza della risposta attesa nel testo dell'utente.
        """
        answer_expected = question["answer"].lower()
        user_text = original_text.lower()
        
        # negation check solo sulla risposta specifica (es: 'it's not 'answer'')
        for token in doc:
            if token.lemma_ in ["not", "no", "never"] and token.dep_ in {"advmod", "neg"}:
                # se la negazione è legata direttamente alla risposta attesa o al suo genitore
                if token.head.text == answer_expected:
                    return False
                
                for child in token.head.children:
                    if child.text == answer_expected:
                        return False

        # utilizzo delle word boundaries (\b) per evitare match parziali (es. "68" in "1968")
        answer_found = re.search(r'\b' + re.escape(answer_expected) + r'\b', user_text)
        
        return True if answer_found else False

    def _analyze_boolean(self, doc, question):
        """Analizza risposte 'sì/no' e rileva conflitti"""
        # determina se la risposta corretta nel DB è true o false
        expected_true = str(question["answer"]).lower() == "true"
        text = doc.text.lower()
        
        has_negation = any(t.dep_ in {"advmod", "neg"} and t.lemma_ in ["not", "no", "never"] for t in doc) or "no" in text or "false" in text
        has_affirmation = any(t.lemma_ in ["yes", "yeah", "sure", "true", "correct", "right", "exactly", "of course"] for t in doc) or "yes" in text or "true" in text

        # 1. Ambiguità : l'utente dice sia sì che no
        if has_negation and has_affirmation:
            return "AMBIGUOUS"

        # 2. Estrazione dell'intenzione
        user_intent = None
        if has_negation:
            user_intent = False
        elif has_affirmation:
            user_intent = True
        
        # se non viene rilevato nè sì nè no, l'input è considerato non valido
        if user_intent is None:
            return False

        # 3. Verifica: l'intento dell'utente coincide con la risposta corretta?
        return user_intent == expected_true

    def _analyze_multiple(self, text, question):
        """Analizza domande a risposta multipla e gestisce Noise Detection"""
        targets = list(question["answer"])
        text = text.lower()
        
        found_now = []
        words_in_targets = 0
        
        # 1. Trova QUALI risposte attese sono presenti nel testo
        for answer in targets:
            if re.search(r'\b' + re.escape(answer.lower()) + r'\b', text):
                found_now.append(answer)
                # conta quante parole compongono la risposta trovata
                words_in_targets += len(answer.split())
        
        # 2. Noise Detection
        if found_now:
            user_words = re.findall(r'\b\w+\b', text)
            noise_words = len(user_words) - words_in_targets
            
            # se le parole extra sono più del doppio delle parole utili, si segnala ambiguità
            if noise_words > (words_in_targets * 2) and len(user_words) > 5:
                return "AMBIGUOUS"
        
        return found_now

