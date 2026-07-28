import json
import re
from nlu import NLUEngine

class DialogueManager:
    """
    Gestisce il flusso del colloquio, lo scoring e la memoria di sessione.
    Coordina NLU e prepara i dati per NLG.
    """

    def __init__(self):
        # caricamento dataset delle domande
        with open('questions.json', 'r', encoding='utf-8') as f:
            self.questions = json.load(f)

        # inizializzazione NLU
        self.nlu = NLUEngine()

        # 1. SESSION STATE: stato attuale del colloquio in corso
        self.session_state = {
            "name": None,
            "score": 0,
            "status": "INTRO"  # stati: INTRO, QUIZ, END
        }

        # 2. MEMORY: memoria storica per analisi comportamentale e varietà del linguaggio
        self.memory = {
            "history": [], # contiene domanda, topic, risposta utente, esito
            "used_templates": set(), # per evitare la ripetizione delle frasi
        }

        # puntatore alla domanda corrente
        self.current_question_index = 0

    def get_status(self):
        return self.session_state["status"]

    def process_input(self, user_input):
        """
        Smista l'input dell'utente in base allo stato corrente
        """
        current_status = self.session_state["status"]

        if current_status == "INTRO":
            return self._handle_intro(user_input)

        elif current_status == "QUIZ":
            return self._handle_quiz(user_input)

        elif current_status == "END":
            return {"type": "END", "msg": "The interview has concluded."}

    def _handle_intro(self, text):
        """Gestisce la fase di benvenuto ed estrazione del nome utente"""
        clean_text = text.strip()
        doc = self.nlu.nlp(clean_text.title())
        
        detected_name = None

        # 1. Pattern Matching esplicito
        patterns = [
            r"(?:my name is|i'm|i am|call me|this is) ([a-zA-Z]+)",
            r"(?:name's) ([a-zA-Z]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                detected_name = match.group(1).title() # capitalizza il risultato (es. Paul)
                break

        # 2. Named Entity Recognition (NER) per trovare entità 'PERSON'
        if not detected_name:
            # raccolte TUTTE le persone trovate, non solo la prima
            found_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            
            if found_names:
                # filtro di sicurezza: rimuove "Lara" o "Croft" se l'utente le ha nominate
                # e ignoriamo nomi troppo corti (probabilmente errori tipo "Su", "Al")
                valid_names = [
                    name for name in found_names 
                    if "lara" not in name.lower() 
                    and "croft" not in name.lower()
                ]
                
                if valid_names:
                    # si prende l'ultimo nome o l'unico valido
                    detected_name = valid_names[-1]

        # gestione caso nome non trovato: assegnazione placeholder
        name_found = True
        if not detected_name:
            detected_name = "Candidate"
            name_found = False

        # aggiornamento stato: si passa alla prima domanda del quiz
        self.session_state["name"] = detected_name
        self.session_state["status"] = "QUIZ"

        return {
            "type": "INTRO_DONE",
            "user_name": detected_name,
            "name_found": name_found,
            "next_question": self.questions[0]["question"]
        }

    def _handle_quiz(self, user_text):
        """
        Gestisce l'interazione logica del quiz: 
        aggiorna score, gestisce ambiguità, avanza domanda
        """

        current_question = self.questions[self.current_question_index]
        question_type = current_question["type"]

        current_found = None
        current_total = 0

        # A. Blocco MULTIPLE

        # setup della memoria
        if question_type == "multiple" and "found_answers" not in self.session_state:
            self.session_state["found_answers"] = [] # accumula le risposte corrette
            self.session_state["expected_answers"] = len(current_question["answer"]) # risposte da trovare

        # chiamata alla NLU
        nlu_result = self.nlu.analyze_answer(user_text, current_question)
        advance_question = False
        dialogue_outcome = "ANSWER_WRONG"

        # elaborazione dei risultati
        if question_type == "multiple":

            # 1. Gestione ambiguità (Noise Detection)
            if nlu_result == "AMBIGUOUS":
                self.session_state["ambiguity_count"] = self.session_state.get("ambiguity_count", 0) + 1

                # la risposta è considerata errata se è ambigua per 3 volte consecutive
                if self.session_state["ambiguity_count"] > 2:
                    dialogue_outcome = "ANSWER_WRONG"
                    advance_question = True
                    self.session_state["ambiguity_count"] = 0
                else:
                    dialogue_outcome = "ANSWER_AMBIGUOUS_MULTIPLE"
                    advance_question = False
            
            # 2. Gestione risposte valide
            else:
                # recupero delle risposte dalla NLU
                identified_answers = nlu_result if isinstance(nlu_result, list) else []
                new_answers = [ # solo quelle che NON sono state già dette
                    a for a in identified_answers
                    if a not in self.session_state["found_answers"]
                ]

                # CASO A: l'utente ha dato almeno una risposta NUOVA
                if new_answers:
                    points_per_item = int(
                        current_question["score"] / self.session_state["expected_answers"]
                    )
                    self.session_state["score"] += (points_per_item * len(new_answers))

                    # flag per capire se si sta completando una lista iniziata prima
                    was_completing = len(self.session_state["found_answers"]) > 0
                    
                    # si aggiungono le nuove risposte alla memoria temporanea
                    self.session_state["found_answers"].extend(new_answers)

                    if len(self.session_state["found_answers"]) >= self.session_state["expected_answers"]:
                        # sono state date tutte le risposte, si avanza
                        dialogue_outcome = "ANSWER_COMPLETION_SUCCESS" if was_completing else "ANSWER_CORRECT"
                        advance_question = True
                    else:
                        # risposte corrette, ma ne mancano ancora altre
                        dialogue_outcome = "ANSWER_INCOMPLETE"
                        advance_question = False
                
                # CASO B: l'utente ha scritto qualcosa, ma NON ci sono risposte NUOVE
                else:
                    # SOTTOCASO B1: l'utente ha ripetuta una risposta già data (duplicato)
                    if identified_answers:
                        self.session_state["ambiguity_count"] = self.session_state.get("ambiguity_count", 0) + 1
                        if self.session_state["ambiguity_count"] > 2:
                            # troppe ripetizioni -> completamento fallito
                            dialogue_outcome = "ANSWER_COMPLETION_FAILED"
                            advance_question = True
                            self.session_state["ambiguity_count"] = 0
                        else:
                            dialogue_outcome = "ANSWER_INCOMPLETE"
                            advance_question = False
                    
                    # SOTTOCASO B2: nessuna risposta valida trovata nel testo (input errato e non ambiguo)
                    else:                        
                        # se l'utente aveva già indovinato qualcosa in precedenza...
                        if len(self.session_state["found_answers"]) > 0:
                            dialogue_outcome = "ANSWER_COMPLETION_FAILED"
                            advance_question = True
                        # se l'utente non aveva indovinato nulla...
                        else:
                            dialogue_outcome = "ANSWER_WRONG"
                            advance_question = True

                # aggiornamento contatori per il feedback all'utente
                current_found = len(self.session_state.get("found_answers", []))
                current_total = self.session_state.get("expected_answers", 0)

                # pulizia della memoria per passaggio alla domanda successiva
                if advance_question:
                    self.session_state.pop("found_answers", None)
                    self.session_state.pop("expected_answers", None)
                    self.session_state["ambiguity_count"] = 0

        # B. Blocco SINGLE o BOOLEAN
        else:

            # reset visualizzazione per domande singole
            current_found = None
            current_total = None

            # 1. Gestione ambiguità
            if nlu_result == "AMBIGUOUS":
                self.session_state["ambiguity_count"] = self.session_state.get("ambiguity_count", 0) + 1
                if self.session_state["ambiguity_count"] > 2:
                    self.session_state["ambiguity_count"] = 0
                    dialogue_outcome = "ANSWER_WRONG"
                    advance_question = True
                else:
                    dialogue_outcome = "ANSWER_AMBIGUOUS_BOOLEAN"
                    advance_question = False
            
            # 2. Verifica risposta
            else:
                self.session_state["ambiguity_count"] = 0
                is_correct = nlu_result
                self.session_state["score"] += current_question["score"] if is_correct else 0
                dialogue_outcome = "ANSWER_CORRECT" if is_correct else "ANSWER_WRONG"
                advance_question = True

        # salva l'interazione su history per l'analisi del mood e recap finale
        self.memory["history"].append({
            "question": current_question["question"],
            "topic": current_question.get("topic", "general knowledge"),
            "user_answer": user_text,
            "dialogue_outcome": dialogue_outcome
        })

        # avanzamento indice domanda
        if advance_question:
            self.current_question_index += 1

        # check fine quiz
        if self.current_question_index >= len(self.questions):
            self.session_state["status"] = "END"
            return {
                "type": "QUIZ_FINISHED",
                "dialogue_outcome": dialogue_outcome,
                "user_name": self.session_state["name"],
                "final_score": self.session_state["score"],
                "verdict": self._compute_verdict(),
                "details": self._analyze_performance(),
                "current_found": current_found,
                "current_total": current_total
            }

        # calcola il mood basandosi sulla storia recente
        current_mood = self._analyze_mood()

        # output per NLG
        return {
            "type": "ANSWER_FEEDBACK",
            "dialogue_outcome": dialogue_outcome,
            "user_name": self.session_state["name"],
            "mood": current_mood,
            "used_templates": self.memory["used_templates"],
            "current_found": current_found,
            "current_total": current_total,
            "next_question": self.questions[self.current_question_index]["question"] if advance_question else None
        }

    def _analyze_mood(self):
        """Analizza la storia per determinare il mood di Lara"""
        history = self.memory["history"]
        if len(history) < 2:
            return "NEUTRAL"

        # estrae gli ultimi due esiti memorizzati
        dialogue_last_outcomes = [h["dialogue_outcome"] for h in history[-2:]]

        # POSITIVE: 2 corrette di fila
        if all(e in ["ANSWER_CORRECT", "ANSWER_COMPLETION_SUCCESS"] for e in dialogue_last_outcomes):
            return "POSITIVE"

        # NEGATIVE: 2 errate di fila
        if all(e in ["ANSWER_WRONG", "ANSWER_COMPLETION_FAILED"] for e in dialogue_last_outcomes):
            return "NEGATIVE"

        return "NEUTRAL"

    def _analyze_performance(self):
        """
        Analisi della sessione per generare un feedback basato su best e worst topic
        """
        history = self.memory["history"]
        errors = [h for h in history if h["dialogue_outcome"] in ["ANSWER_WRONG", "ANSWER_COMPLETION_FAILED"]]
        successes = [h for h in history if h["dialogue_outcome"] in ["ANSWER_CORRECT", "ANSWER_COMPLETION_SUCCESS"]]

        # mapping per tradurre i tag JSON in descrizioni per Lara
        topic_descriptions = {
            "biography": "my personal life and family",
            "artifacts": "ancient and dangerous relics",
            "adventures": "my past expeditions",
            "general knowledge": "basic facts"
        }

        # 1. Calcolo topic peggiore
        worst_topic = "nothing"
        if errors:
            counts = {}
            for e in errors:
                t = e.get("topic", "general knowledge")
                counts[t] = counts.get(t, 0) + 1
            # prende il topic con più errori
            worst_topic_key = max(counts, key=counts.get)
            worst_topic = topic_descriptions.get(worst_topic_key, "some fundamental details")

        # 2. Calcolo topic migliore (basato sull'ultimo successo ottenuto)
        best_topic = "no specific subject"
        if successes:
            best_topic_key = successes[-1].get("topic", "general knowledge")
            best_topic = topic_descriptions.get(best_topic_key, "my history")

        return {
            "worst_topic": worst_topic,
            "best_topic": best_topic,
            "error_count": len(errors)
        }

    def _compute_verdict(self):
        """Genera verdetto finale basato sullo score ratio"""
        max_score = sum(q["score"] for q in self.questions)
        my_score = self.session_state["score"]
        ratio = my_score / max_score

        # 100% (Perfect), >= 60% (Passed), < 60% (Failed)
        if ratio == 1.0:
            return "PERFECT"
        elif ratio >= 0.6:
            return "PASSED"
        else:
            return "FAILED"

