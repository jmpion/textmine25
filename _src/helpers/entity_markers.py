ET_TO_EXHAUSTIVE_ET = {
    'MILITARY_ORGANISATION': ['MILITARY_ORGANISATION', 'GOVERNMENT_ORGANISATION', 'ORGANISATION', 'ACTOR'],
    'NON_MILITARY_GOVERNMENT_ORGANISATION': ['NON_MILITARY_GOVERNMENT_ORGANISATION', 'GOVERNMENT_ORGANISATION', 'ORGANISATION', 'ACTOR'],
    'GROUP_OF_INDIVIDUALS': ['GROUP_OF_INDIVIDUALS', 'ORGANISATION', 'ACTOR'],
    'INTERGOVERNMENTAL_ORGANISATION': ['INTERGOVERNMENTAL_ORGANISATION', 'ORGANISATION', 'ACTOR'],
    'NON_GOVERNMENTAL_ORGANISATION': ['NON_GOVERNMENTAL_ORGANISATION', 'ORGANISATION', 'ACTOR'],
    'CIVILIAN': ['CIVILIAN', 'PERSON', 'ACTOR'],
    'TERRORIST_OR_CRIMINAL': ['TERRORIST_OR_CRIMINAL', 'PERSON', 'ACTOR'],
    'MILITARY': ['MILITARY', 'PERSON', 'ACTOR'],
    'ACCIDENT': ['ACCIDENT', 'EVENT'],
    'CBRN_EVENT': ['CBRN_EVENT', 'EVENT'],
    'AGITATING_TROUBLE_MAKING': ['AGITATING_TROUBLE_MAKING', 'CIVIL_UNREST', 'EVENT'],
    'CIVIL_WAR_OUTBREAK': ['CIVIL_WAR_OUTBREAK', 'CIVIL_UNREST', 'EVENT'],
    'COUP_D_ETAT': ['COUP_D_ETAT', 'CIVIL_UNREST', 'EVENT'],
    'DEMONSTRATION': ['DEMONSTRATION', 'CIVIL_UNREST', 'EVENT'],
    'ELECTION': ['ELECTION', 'CIVIL_UNREST', 'EVENT'],
    'GATHERING': ['GATHERING', 'CIVIL_UNREST', 'EVENT'],
    'ILLEGAL_CIVIL_DEMONSTRATION': ['ILLEGAL_CIVIL_DEMONSTRATION', 'CIVIL_UNREST', 'EVENT'],
    'NATURAL_CAUSES_DEATH': ['NATURAL_CAUSES_DEATH', 'CIVIL_UNREST', 'EVENT'],
    'RIOT': ['RIOT', 'CIVIL_UNREST', 'EVENT'],
    'STRIKE': ['STRIKE', 'CIVIL_UNREST', 'EVENT'],
    'SUICIDE': ['SUICIDE', 'CIVIL_UNREST', 'EVENT'],
    'BOMBING': ['BOMBING', 'CRIMINAL_EVENT', 'EVENT'],
    'CRIMINAL_ARREST': ['CRIMINAL_ARREST', 'CRIMINAL_EVENT', 'EVENT'],
    'DRUG_OPERATION': ['DRUG_OPERATION', 'CRIMINAL_EVENT', 'EVENT'],
    'HOOLIGANISM_TROUBLEMAKING': ['HOOLIGANISM_TROUBLEMAKING', 'CRIMINAL_EVENT', 'EVENT'],
    'POLITICAL_VIOLENCE': ['POLITICAL_VIOLENCE', 'CRIMINAL_EVENT', 'EVENT'],
    'THEFT': ['THEFT', 'CRIMINAL_EVENT', 'EVENT'],
    'TRAFFICKING': ['TRAFFICKING', 'CRIMINAL_EVENT', 'EVENT'],
    'ECONOMICAL_CRISIS': ['ECONOMICAL_CRISIS', 'LARGE_SCALE_EVENT', 'EVENT'],
    'EPIDEMIC': ['EPIDEMIC', 'LARGE_SCALE_EVENT', 'EVENT'],
    'FIRE': ['FIRE', 'LARGE_SCALE_EVENT', 'EVENT'],
    'NATURAL_EVENT': ['NATURAL_EVENT', 'LARGE_SCALE_EVENT', 'EVENT'],
    'POLLUTION': ['POLLUTION', 'LARGE_SCALE_EVENT', 'EVENT'],
    'MATERIEL': ['MATERIEL'],
    'PLACE': ['PLACE'],
    'CATEGORY': ['CATEGORY'],
    'COLOR': ['COLOR'],
    'HEIGHT': ['HEIGHT'],
    'LENGTH': ['LENGTH'],
    'MATERIAL_REFERENCE': ['MATERIAL_REFERENCE'],
    'NATIONALITY': ['NATIONALITY'],
    'QUANTITY_EXACT': ['QUANTITY_EXACT', 'QUANTITY'],
    'QUANTITY_FUZZY': ['QUANTITY_FUZZY', 'QUANTITY'],
    'QUANTITY_MAX': ['QUANTITY_MAX', 'QUANTITY'],
    'QUANTITY_MIN': ['QUANTITY_MIN', 'QUANTITY'],
    'TIME_EXACT': ['TIME_EXACT', 'TIME'],
    'TIME_FUZZY': ['TIME_FUZZY', 'TIME'],
    'TIME_MAX': ['TIME_MAX', 'TIME'],
    'TIME_MIN': ['TIME_MIN', 'TIME'],
    'WIDTH': ['WIDTH'],
    'WEIGHT': ['WEIGHT'],
    'LATITUDE': ['LATITUDE'],
    'LONGITUDE': ['LONGITUDE'],
}

from .constants import CONFIG
from .filter import ALL_ET

def add_entity_markers(text, entities):
    """
    Ajoute des tokens spéciaux autour des entités dans le texte.
    
    Args:
    - text: str, le texte brut.
    - entities: list of dict, une liste d'entités où chaque entité est représentée 
                par un dictionnaire avec les clés 'start', 'end', 'entity', 'type'.
    - CONFIG: dict, CONFIGuration du mode de wrapping des entités.
    - ET_TO_EXHAUSTIVE_ET: dict, mappage des entités vers leurs types exhaustifs.
    
    Returns:
    - str: le texte avec les tokens spéciaux pour les entités.
    """
    entities_flat = []

    # Combine mentions from both entities and assign their roles (S for subject, O for object)
    for m in entities[0]['mentions']:
        m_tmp = m.copy()
        m_tmp['type'] = entities[0]['type']
        m_tmp['role'] = 'S'
        entities_flat.append(m_tmp)
    for m in entities[1]['mentions']:
        m_tmp = m.copy()
        m_tmp['type'] = entities[1]['type']
        m_tmp['role'] = 'O'
        entities_flat.append(m_tmp)

    # Sort entities by start index (to avoid conflicts when inserting)
    entities_flat.sort(key=lambda x: x['start'])
    
    # Determine the wrapper mode from the CONFIG
    wrapper_mode = CONFIG['entity_markers']['wrapper_mode']

    # Store the markers (start and end) to be inserted into a list with their positions
    insertions = []

    for entity in entities_flat:
        ent_type, ent_role = entity['type'], entity['role']
        
        # Handling 'Complex', 'Simple', and 'High-level' wrapper modes
        if wrapper_mode == 'Complex':
            # Exhaustive entity type, involving multiple levels of entity markers
            exhaustive_ent_types = ET_TO_EXHAUSTIVE_ET[ent_type]
            for ent_type_tmp in exhaustive_ent_types:
                start_token = f"[{ent_role}-{ent_type_tmp}]"
                end_token = f"[/{ent_role}-{ent_type_tmp}]"
                insertions.append((entity['start'], 'start', start_token))
                insertions.append((entity['end'], 'end', end_token))
        
        elif wrapper_mode == 'Simple':
            # Simple entity wrapping using only the specific type
            start_token = f"[{ent_role}-{ent_type}]"
            end_token = f"[/{ent_role}-{ent_type}]"
            insertions.append((entity['start'], 'start', start_token))
            insertions.append((entity['end'], 'end', end_token))

        elif wrapper_mode == "High-level":
            # Only wrap with the highest-level entity type
            exhaustive_ent_types = ET_TO_EXHAUSTIVE_ET[ent_type]
            high_level_ent_type = exhaustive_ent_types[-1]  # Use the highest level
            start_token = f"[{ent_role}-{high_level_ent_type}]"
            end_token = f"[/{ent_role}-{high_level_ent_type}]"
            insertions.append((entity['start'], 'start', start_token))
            insertions.append((entity['end'], 'end', end_token))

    # Sort insertions: start markers come before end markers at the same position
    insertions.sort(key=lambda x: (x[0], 1 if x[1] == 'end' else 0))

    # Adjust positions and insert the markers into the text
    offset = 0  # Initialisation du décalage
    insert_positions = {}
    for idx, marker_type, token in insertions:
        insert_position = idx + offset
        text = text[:insert_position] + token + text[insert_position:]
        offset += len(token)  # Update the offset after insertion
        insert_positions[insert_position] = token

    return text, insert_positions


def additional_tokens_from_wrapper_mode(wrapper_mode: str):
    """Returns the additional tokens to add to the tokenizer, based on the wrapper mode selected for marking the entities.

    Args:
        wrapper_mode (str): how to wrap entities in the text. 'Simple' for using low-level entity type. 'Complex' for using all entity types for each entity. 'High-level' for using high-level entity types only.

    Returns:
        _type_: the additonal special tokens to add to the tokenizer of the experiment, due to the entity wrapping.
    """
    assert wrapper_mode in ['Complex', 'High-level', 'Simple']
    added_special_tokens = []
    if wrapper_mode=='Complex':
        for ent_type in ALL_ET:
            seen_ent_type = []
            exh_ent_type = ET_TO_EXHAUSTIVE_ET[ent_type]
            for ent_type_tmp in exh_ent_type:
                if ent_type_tmp not in seen_ent_type:
                    seen_ent_type.append(ent_type_tmp)
                    for ent_role in ['S', 'O']:
                        start_token = f"[{ent_role}-{ent_type_tmp}]"
                        end_token = f"[/{ent_role}-{ent_type_tmp}]"
                        added_special_tokens.append(start_token)
                        added_special_tokens.append(end_token)
    elif wrapper_mode=='Simple':
        for ent_type in ALL_ET:
            for ent_role in ['S', 'O']:
                start_token = f"[{ent_role}-{ent_type}]"
                end_token = f"[/{ent_role}-{ent_type}]"
                added_special_tokens.append(start_token)
                added_special_tokens.append(end_token)
    elif wrapper_mode=='High-level':
        seen_ent_type = []
        for ent_type in ALL_ET:
            exh_ent_type = ET_TO_EXHAUSTIVE_ET[ent_type]
            ent_type_tmp = exh_ent_type[-1]
            if ent_type_tmp not in seen_ent_type:
                seen_ent_type.append(ent_type_tmp)
                for ent_role in ['S', 'O']:
                    start_token = f"[{ent_role}-{ent_type_tmp}]"
                    end_token = f"[/{ent_role}-{ent_type_tmp}]"
                    added_special_tokens.append(start_token)
                    added_special_tokens.append(end_token)
    return added_special_tokens

def special_tokens_from_wrapper_mode(wrapper_mode: str, ent_type_subject: str, ent_type_object: str):
    assert wrapper_mode in ['Complex', 'High-level', 'Simple']
    if wrapper_mode=='Simple':
        pass
    elif wrapper_mode=='High-level':            
        exh_ent_type_subject = ET_TO_EXHAUSTIVE_ET[ent_type_subject]
        exh_ent_type_object = ET_TO_EXHAUSTIVE_ET[ent_type_object]
        ent_type_subject = exh_ent_type_subject[-1]
        ent_type_object = exh_ent_type_object[-1]
    special_tokens = {
        'special_token_subject_start': f"[S-{ent_type_subject}]",
        'special_token_object_start': f"[O-{ent_type_object}]",
        'special_token_subject_end': f"[/S-{ent_type_subject}]",
        'special_token_object_end': f"[/O-{ent_type_object}]",
        }
    return special_tokens