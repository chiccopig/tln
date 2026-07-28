import os
from dm import DialogueManager
from nlg import NLGEngine

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """
    Coordina il flusso di dati tra l'utente, la logica del dialogo e la generazione del linguaggio
    """
    
    # Engine Initialization
    dm = DialogueManager() # gestisce lo stato e la logica del quiz
    nlg = NLGEngine() # trasforma i dati logici in frasi naturali

    clear_console()

    # FASE 1: Introduzione
    # messaggio di benvenuto statico
    print("LARA: Welcome. I'm Lara Croft. Many aspire to be my assistants,")
    print("      but few possess the necessary preparation. Let's start with your name.")
    
    # loop di validazione: impedisce input vuoti per il nome
    user_input = input("YOU: ")
    while not user_input.strip():
        user_input = input("LARA: Don't be shy. What is your name? \nYOU: ")
    
    # il dm analizza il nome (tramite NLU interna)
    intro_data = dm.process_input(user_input)
    
    # NLG genera il saluto personalizzato
    intro_response, _ = nlg.generate_response(intro_data)
    print(f"\n{intro_response}")
    
    # FASE 2: Loop del Quiz
    # il ciclo continua finché lo stato interno del dm è 'QUIZ'
    while dm.get_status() == "QUIZ":
        user_input = input("YOU: ")
        
        # gestione input vuoto
        if not user_input.strip():
            print("LARA: Silence won't help you. Answer my question.")
            continue
            
        # 1. Analisi: il dm processa l'input e determina l'esito (Corretto/Errato/Ambiguo)
        dialogue_data = dm.process_input(user_input)

        # 2. Sincronizzazione memoria: passaggio dei template già usati al generatore
        dialogue_data["used_templates"] = dm.memory["used_templates"]
        
        # 3. Generazione: NLG crea la risposta basandosi sul mood e sull'esito
        full_response, atomic_feedback = nlg.generate_response(dialogue_data)
        
        # 4. Aggiornamento memoria: salvataggio del feedback in memoria (SOLO template atomico)
        if "used_templates" in dm.memory:
            dm.memory["used_templates"].add(atomic_feedback)
        
        # stampa della risposta finale
        print(f"\n{full_response}")
        
    # FASE 3: Conclusione
    # quando lo stato cambia in 'END', il loop si interrompe
    # il verdetto finale viene stampato nell'ultima iterazione del loop sopra
    print("\n--- End of Session ---")

if __name__ == "__main__":
    main()

