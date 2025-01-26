from helpers.constants import RELATIONS_DEFINITIONS

def ask_yes_no_question(question: str, tokenizer=None, model=None, mode='openai', client=None) -> str:
    """Generates the answer from the LLM to the given question.
    Args:
        question (str): Question for the LLM.
    Returns:
        str: Answer from the LLM.
    """
    prompt = f"Q: {question}\nA:"
    if mode=='huggingface':
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=3)
        new_tokens = output[0][inputs['input_ids'].shape[-1]:]  # Slice out the new tokens
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    elif mode=='openai':
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Default: "gpt-4o-mini"
            messages=[{"role": "user", "content": prompt}],
            seed=0, # to try and have reproducible results.
        )
        answer = completion.choices[0].message.content
    return answer

def validate_relation_llm_yn(text, es_all, eo_all, r, tokenizer=None, model=None, mode='openai', client=None):
    if r == 'INJURED_NUMBER':
        question = f"Voici un rapport de renseignement:\n\n'{text}'\n\nDans ce rapport de renseignement, la quantité '{eo_all['mentions'][0]}' indique-t-elle le nombre de personnes ayant été blessées non-mortellement lors de l'événement '{es_all['mentions'][0]}' ? Réponds simplement Oui ou Non. À noter : si le texte n'indique pas explicitement qu'il y a des blessures, alors la réponse sera Non. De même, même si des gens sont hospitalisés, qu'on leur porte secours, ou qu'on les qualifie de victimes, cela ne signifie pas nécessairement qu'ils sont blessés. Tu ne dois pas deviner les blessures, tu dois les trouver indiquées explicitement dans le rapport."
    else:
        question = f"""On définit la relation '{r}' comme étant '{RELATIONS_DEFINITIONS[r]}'.

Voici un rapport de renseignement que l'on souhaite étudier : {text}

Dans ce rapport de renseignement, l'entité sujet [{es_all['mentions'][0]}], de type {es_all['type']}, et l'entité objet [{eo_all['mentions'][0]}, de type {eo_all['type']} sont-elles reliées par la relation '{r}' ? Réponds simplement Oui ou Non."""
    answer = ask_yes_no_question(question, tokenizer=tokenizer, model=model, mode=mode, client=client)
    short_answer = answer.strip().split(' ')[0]
    if short_answer[:3] in ["Oui", "Yes"]:
        return True
    elif short_answer[:2] in ["No"]:
        return False
    else:
        print(f"verbose answer: {answer}", flush=True)
        return False