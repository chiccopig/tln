import random
from simplenlg import Lexicon, NLGFactory, Realiser
from simplenlg.features import Feature, Tense, NumberAgreement

class NLGEngine:
    """
    Genera le risposte di Lara combinando:
    - template predefiniti
    - frasi dinamiche realizzate con SimpleNLG
    """

    def __init__(self):

        # inizializzazione SimpleNLG
        self.lexicon = Lexicon.getDefaultLexicon()
        self.factory = NLGFactory(self.lexicon)
        self.realiser = Realiser(self.lexicon)

        # templates per mantenere il tono di Lara
        self.templates = {
            "INTRO_REPLY_KNOWN": [
                "Very well, {name}. Let's see if you are as useful as you claim.",
                "Right, {name}. But I warn you: I have no time for incompetence.",
                "An interesting name, {name}. I hope your skills live up to it."
            ],
            "INTRO_REPLY_UNKNOWN": [
                "A name would have been useful. For now, you are '{name}'.",
                "Few words, I see. Very well, '{name}', let's see what you can do.",
                "You didn't introduce yourself. I'll settle for this: {name}.",
            ],
            "FEEDBACK_CORRECT": {
                "NEUTRAL": [
                    "Exactly. Just as I remember it.",
                    "Correct. Perhaps there's hope for you after all.",
                    "Good. You didn't let yourself be fooled."
                ],
                "POSITIVE": [
                    "Impressive, you're a true expert on my history!",
                    "Excellent! You surprise me; I wouldn't have bet on you.",
                    "Not bad at all! Keep this up and I might actually trust you.",
                    "Outstanding! You're proving to be much more than just a lucky guesser."
                ]
            },
            "FEEDBACK_WRONG": {
                "NEUTRAL": [
                    "No. Disappointing.",
                    "Incorrect. You should study my journals more closely.",
                    "Absolutely not. Focus, {name}."
                ],
                "NEGATIVE": [
                    "Another mistake? I'm starting to lose my patience.",
                    "This is embarrassing... do you know anything about me at all?",
                    "Another failure. If we were in the field, you'd be in serious trouble by now.",
                    "Are you just guessing? It shows.",
                    "That's enough, {name}. Focus or we're done here.",
                    "My father would never have hired someone so ill-prepared."
                ]
            },
            "FEEDBACK_INCOMPLETE": [
                "That's not all. Keep going.",
                "Something is still missing. Think harder.",
                "You know there's more to it. Say it.",
                "Interesting... but incomplete."
            ],
            "FEEDBACK_COMPLETION_SUCCESS": [
                "Finally, you've completed the list. That wasn't so hard, was it?",
                "You've filled the gaps. Move on.",
                "You've added the missing pieces. Well done."
            ],
            "FEEDBACK_COMPLETION_FAILED": [
                "No, that has nothing to do with it. Time's up.",
                "Wrong. I hoped the hint would help you, but I was mistaken.",
                "Enough. That was your last chance."
            ],
            "FEEDBACK_AMBIGUOUS_BOOLEAN": [
                "Make up your mind. Is it a yes or a no? I can't stand indecision.",
                "First you affirm, then you deny... Be consistent, Candidate.",
                "It's a simple question: true or false? Stop contradicting yourself."
            ],
            "FEEDBACK_AMBIGUOUS_MULTIPLE": [
                "Too many useless words. Just tell me what I asked.",
                "I don't have time for this chatter. List the items clearly.",
                "I asked for a list, not a monologue. Try again, with less noise.",
                "Be concise. I'm not interested in your comments, only the facts."
            ],
            "TRANSITION": [
                "Next question: ",
                "Tell me: ",
                "Listen closely: ",
                "Moving on: "
            ],
            "VERDICT_PERFECT": [
                "Incredible. You didn't miss a beat. Pack your bags, {name}, we leave at dawn. You're hired.",
                "Exceptional. It's rare to find someone on my level. Welcome to the team."
            ],
            "VERDICT_PASSED": [
                "Not bad, {name}. You still have much to learn, but I see potential. You're on probation.",
                "Sufficient. I'll take you with me, but try not to slow me down."
            ],
            "VERDICT_FAILED": [
                "As I suspected. You're not ready for this lifestyle. Goodbye, {name}.",
                "Disastrous. Come back when you've actually read a history book. This interview is over."
            ]
        }

    def _realise_completion_progress(self, found, total):
        """Genera il feedback parziale per le domande multiple"""
        clause = self.factory.createClause()
        clause.setSubject("You")
        clause.setVerb("identify")
        
        # Present Perfect: 'have identified'
        clause.setFeature(Feature.TENSE, Tense.PRESENT)
        clause.setFeature(Feature.PERFECT, True)
        clause.setFeature(Feature.NUMBER, NumberAgreement.PLURAL)

        # specifica la quantità come specifier del sostantivo
        quantity_text = f"{found} out of {total}"
        
        # creazione del sintagma nominale (noun phrase)
        item_np = self.factory.createNounPhrase("item")
        item_np.setSpecifier(quantity_text)

        # forza il plurale se necessario
        item_np.setFeature(Feature.NUMBER, NumberAgreement.PLURAL)
        
        clause.setObject(item_np)

        return self.realiser.realiseSentence(clause)
    
    def _realise_performance_report(self, details, error_count):
        """Genera un riassunto della performance basato su best e worst topic"""
        best_topic = details.get("best_topic", "history")
        worst_topic = details.get("worst_topic", "details")
        
        # coordinatore che gestisce la congiunzione (and/but) tra due proposizioni
        coord = self.factory.createCoordinatedPhrase()

        # 1. Proposizione positiva
        # Se c'è un tema positivo e non è generico
        if best_topic and best_topic != "no specific subject":
            p1 = self.factory.createClause("you", "grasp")
            obj_text = f"the nuances of {best_topic}"
            p1.setObject(obj_text)
            p1.setFeature(Feature.TENSE, Tense.PAST)
            coord.addCoordinate(p1)

        # 2. Proposizione negativa
        if error_count > 0:
            p2 = self.factory.createClause("you", "commit")
            p2.setFeature(Feature.TENSE, Tense.PAST)

            # costruzione dinamica del sintagma nominale per gli errori
            noun_errors = self.factory.createNounPhrase("error")
            noun_errors.setSpecifier(str(error_count))
            
            if error_count > 1:
                noun_errors.setFeature(Feature.NUMBER, NumberAgreement.PLURAL)
            
            p2.setObject(noun_errors)

            # prepositional phrase (pp): aggiunge contesto
            pp = self.factory.createPrepositionPhrase("regarding", worst_topic)
            p2.addComplement(pp)

            coord.addCoordinate(p2)
            
            # in caso di presenza di entrambe le parti, si uniscono con "but"
            if best_topic and best_topic != "no specific subject":
                coord.setConjunction("but")
            else:
                # se l'utente ha solo sbagliato, restituisce solo la parte negativa
                return self.realiser.realiseSentence(p2)

        # se non ci sono errori, restituisce solo la parte positiva
        if error_count == 0 and best_topic:
            return self.realiser.realiseSentence(coord) # ritorna solo p1

        # ritorna la frase combinata "You ... but ..."
        return self.realiser.realiseSentence(coord)
    
    def _realise_score_report(self, points, verdict):
        """Genera il verdetto finale con punteggio e giudizio"""
        clause = self.factory.createClause()
        clause.setSubject("You")
        
        # scelta lessicale dinamica basata sul successo
        if verdict == "PERFECT":
            clause.setVerb("achieve")
        else:
            clause.setVerb("score")
            
        clause.setFeature(Feature.TENSE, Tense.PAST)
        
        # gestione oggetto: "X points" o "only X points"
        noun_points = self.factory.createNounPhrase("point")
        noun_points.setSpecifier(str(points))
        
        # modificatore: se il punteggio è basso
        if points < 15 and verdict != "PERFECT":
            noun_points.addPreModifier("only")
            
        clause.setObject(noun_points)
        
        # crazione di una frase relativa come complemento (which is...)
        relative = self.factory.createClause()
        relative.setSubject("which")
        relative.setVerb("be")
        
        # mappatura del verdetto in descrizioni naturali
        compl = "a perfect result" if verdict == "PERFECT" else "insufficient"
        if verdict == "PASSED":
            compl = "a solid start"
            
        relative.setObject(compl)

        # addComplement attacca la relativa alla frase principale con la punteggiatura corretta
        clause.addComplement(relative)

        return self.realiser.realiseSentence(clause)

    def generate_response(self, dm_data):
        """
        Coordina i template predefiniti con le frasi realizzate da SimpleNLG
        """
        message_type = dm_data.get("type")
        user_name = dm_data.get("user_name", "Candidate")
        used_templates = dm_data.get("used_templates", set())
        
        # logica per feedback dinamico (template + SimpleNLG)
        dynamic_stats = ""
        # se dm comunica un completamento parziale (multiple)
        if dm_data.get("current_found") is not None and dm_data.get("current_total", 0) > 0:
            dynamic_stats = self._realise_completion_progress(
                dm_data["current_found"],
                dm_data["current_total"]
            )

        # 1. Gestione introduzione
        if message_type == "INTRO_DONE":
            # sceglie una frase a seconda se il nome è stato compreso o meno
            option_list = (
                self.templates["INTRO_REPLY_KNOWN"]
                if dm_data.get("name_found")
                else self.templates["INTRO_REPLY_UNKNOWN"]
            )
            intro_reply = self.pick_unique_template(option_list, used_templates).format(name=user_name)
            # ritorna la risposta composta e il template usato (per la memoria)
            return f"LARA: {intro_reply}\n\nLARA: First question. {dm_data.get('next_question')}", intro_reply

        # 2. Gestione feedback durante il quiz
        elif message_type == "ANSWER_FEEDBACK":
            outcome = dm_data["dialogue_outcome"]
            mood = dm_data.get("mood", "NEUTRAL") # mood calcolato dal DM
            
            # selezione del set di template basato su esito e mood
            if outcome == "ANSWER_CORRECT":
                candidate_templates = self.templates["FEEDBACK_CORRECT"].get(mood, self.templates["FEEDBACK_CORRECT"]["NEUTRAL"])
            elif outcome == "ANSWER_WRONG":
                candidate_templates = self.templates["FEEDBACK_WRONG"].get(mood, self.templates["FEEDBACK_WRONG"]["NEUTRAL"])
            elif outcome == "ANSWER_INCOMPLETE":
                candidate_templates = self.templates["FEEDBACK_INCOMPLETE"]
            elif outcome == "ANSWER_COMPLETION_SUCCESS":
                candidate_templates = self.templates["FEEDBACK_COMPLETION_SUCCESS"]
            elif outcome == "ANSWER_COMPLETION_FAILED":
                candidate_templates = self.templates["FEEDBACK_COMPLETION_FAILED"]
            elif outcome == "ANSWER_AMBIGUOUS_BOOLEAN":
                candidate_templates = self.templates["FEEDBACK_AMBIGUOUS_BOOLEAN"]
            elif outcome == "ANSWER_AMBIGUOUS_MULTIPLE":
                candidate_templates = self.templates["FEEDBACK_AMBIGUOUS_MULTIPLE"]
            else:
                candidate_templates = ["I see."]

            # estrazione di un template unico e formattazione con il nome utente
            chosen_template = self.pick_unique_template(candidate_templates, used_templates)
            visible_feedback = chosen_template.format(name=user_name)

            # combina il template (stile) con le stats di SimpleNLG (dati)
            if dynamic_stats:
                visible_feedback = f"{visible_feedback} {dynamic_stats}"

            # gestione della transizione alla prossima domanda (se prevista)
            next_question = dm_data.get("next_question")
            if next_question and outcome not in ["ANSWER_INCOMPLETE", "ANSWER_AMBIGUOUS_MULTIPLE"]:
                # aggiunge una frase di raccordo
                transition = random.choice(self.templates["TRANSITION"])
                return f"LARA: {visible_feedback}\n\nLARA: {transition}{next_question}", chosen_template
            
            return f"LARA: {visible_feedback}", chosen_template

        # 3. Gestione verdetto finale (quiz concluso)
        elif message_type == "QUIZ_FINISHED":
            last_outcome = dm_data.get("dialogue_outcome")

            if last_outcome in ["ANSWER_CORRECT", "ANSWER_COMPLETION_SUCCESS"]:
                candidate_templates = self.templates["FEEDBACK_CORRECT"]["NEUTRAL"]
            elif last_outcome in ["ANSWER_WRONG", "ANSWER_COMPLETION_FAILED"]:
                candidate_templates = self.templates["FEEDBACK_WRONG"]["NEGATIVE"] # Lara è più dura alla fine
            else:
                candidate_templates = ["I see how this ends."]

            last_feedback = random.choice(candidate_templates).format(name=user_name)
            
            if dynamic_stats:
                last_feedback = f"{last_feedback} {dynamic_stats}"

            # assemblaggio del report finale usando simpleNLG
            details = dm_data.get("details", {})
            error_count = details.get("error_count", 0)
            
            # generazione frasi complesse (performance e punteggio)
            performance_report = self._realise_performance_report(details, error_count)

            verdict = dm_data["verdict"]
            points = dm_data["final_score"]
            score_report = self._realise_score_report(points, verdict)

            # scelta della frase di addio basata sul verdetto (PERFECT, PASSED, FAILED)
            verdict_options = self.templates.get(f"VERDICT_{verdict}", self.templates["VERDICT_FAILED"])
            final_sentence = random.choice(verdict_options).format(name=user_name)
            
            # composizione del blocco di testo finale multiline
            full_response = (
                f"LARA: {last_feedback}\n"
                f"LARA: {performance_report}\n"
                f"LARA: {score_report}\n\n"
                f"LARA: {final_sentence}"
            )
            
            return full_response, last_feedback
            
        return "End of session.", "End"
    
    def pick_unique_template(self, candidate_templates, used_templates):
        """Assicura che Lara non dica la stessa cosa finché ha alternative disponibili"""
        # sottrae dall'elenco delle opzioni quelle già presenti nella memoria 'used_templates'
        available = [t for t in candidate_templates if t not in used_templates]
        
        # se esistono frasi mai usate, ne sceglie una
        if available:
            return random.choice(available)
        
        # se tutte le opzioni sono state usate, resetta parzialmente la memoria per poterle riutilizzare
        else:
            for t in candidate_templates:
                if t in used_templates:
                    used_templates.remove(t)
            return random.choice(candidate_templates)

