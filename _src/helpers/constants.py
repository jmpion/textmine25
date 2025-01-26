import yaml
from .filter import (ORGANISATION_ET, PERSON_ET, ACTOR_ET, EVENT_ET, MATERIAL_ET, PLACE_ET, NATIONALITY_ET, QUANTITY_ET, TIME_ET, COLOR_ET, HEIGHT_ET, LENGTH_ET, WIDTH_ET, MATERIAL_REFERENCE_ET, WEIGHT_ET, CATEGORY_ET, LATITUDE_ET, LONGITUDE_ET)

# Configuration.
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

# Validity checks on the CONFIG.
print(CONFIG)
assert CONFIG['one_class']['support'] in ['Yes', 'No']
assert CONFIG['submission_type'] in ['VALIDATION', 'TEST']
assert CONFIG['entity_markers']['wrapper_mode'] in ['Simple', 'Complex', 'High-level']
assert CONFIG['one_class']['one_class_on'] in ['Yes', 'No']
assert CONFIG['llm_yn']['train_or_test'] in ['Train', 'Test']
assert CONFIG['model']['class_weights'] in ['default', 'balanced']
assert not (CONFIG['separate_spans_embeddings']=='Yes' and CONFIG['entity_markers']['entity_markers_on']=='Yes' and CONFIG['extraction_mode']=='all tokens'), "Currently, using separate spans embeddings is incompatible with using entity markers and extracting all entity tokens."
assert CONFIG['extraction_mode'] in ['all_tokens', 'special_tokens']

# Relation types ids.
RELATIONS_IDS = {
    "IS_LOCATED_IN": 1,
    "IS_OF_NATIONALITY": 2,
    "CREATED": 3,
    "HAS_CONTROL_OVER": 4,
    "INITIATED": 5,
    "IS_AT_ODDS_WITH": 6,
    "IS_COOPERATING_WITH": 7,
    "IS_IN_CONTACT_WITH": 8,
    "IS_PART_OF": 9,
    "DEATHS_NUMBER": 10,
    "END_DATE": 11,
    "HAS_CONSEQUENCE": 12,
    "INJURED_NUMBER": 13,
    "START_DATE": 14,
    "STARTED_IN": 15,
    "HAS_COLOR": 16,
    "HAS_FOR_HEIGHT": 17,
    "HAS_FOR_LENGTH": 18,
    "HAS_FOR_WIDTH": 19,
    "HAS_QUANTITY": 20,
    "IS_REGISTERED_AS": 21,
    "WEIGHS": 22,
    "WAS_CREATED_IN": 23,
    "WAS_DISSOLVED_IN": 24,
    "IS_OF_SIZE": 25,
    "OPERATES_IN": 26,
    "DIED_IN": 27,
    "HAS_CATEGORY": 28,
    "HAS_FAMILY_RELATIONSHIP": 29,
    "GENDER_FEMALE": 30,
    "GENDER_MALE": 31,
    "IS_BORN_IN": 32,
    "IS_BORN_ON": 33,
    "IS_DEAD_ON": 34,
    "RESIDES_IN": 35,
    "HAS_LATITUDE": 36,
    "HAS_LONGITUDE": 37,
}

ID_TO_RELATION = {value: key for key, value in RELATIONS_IDS.items()}

ENT_TYPES = ORGANISATION_ET + PERSON_ET + ACTOR_ET + EVENT_ET + MATERIAL_ET + PLACE_ET + NATIONALITY_ET + QUANTITY_ET + TIME_ET + COLOR_ET + HEIGHT_ET + LENGTH_ET + WIDTH_ET + MATERIAL_REFERENCE_ET + WEIGHT_ET + CATEGORY_ET + LATITUDE_ET + LONGITUDE_ET

RELATIONS_DEFINITIONS = {
    "IS_LOCATED_IN": "The Subject's location",
    "IS_OF_NATIONALITY": "The Subject's nationality",
    "CREATED": "The Subject played a role in the creation of the Object",
    "HAS_CONTROL_OVER": "The Subject has control over Object",
    "INITIATED": "The Object has been created, organised, or is caused by the Subject",
    "IS_AT_ODDS_WITH": "The Subject and the Object have opposed goals and are at odds",
    "IS_COOPERATING_WITH": "The Subject works for / cooperates with the Object without being part of if the object is an Organization",
    "IS_IN_CONTACT_WITH": "The Subject and the Object have been in contact",
    "IS_PART_OF": "The Subject is part of the Object",
    "DEATHS_NUMBER": "The Number of people who died because of the event",
    "END_DATE": "The Subject's end date",
    "HAS_CONSEQUENCE": "The Subject is a trigger for the Object",
    "INJURED_NUMBER": "The number of people with non-fatal injuries in the event",
    "START_DATE": "The Subject's start date",
    "STARTED_IN": "The Subject's start location",
    "HAS_COLOR": "The Subject's colors",
    "HAS_FOR_HEIGHT": "The Subject's height",
    "HAS_FOR_LENGTH": "The Subject's length",
    "HAS_FOR_WIDTH": "The Subject's width",
    "HAS_QUANTITY": "The number of unit items",
    "IS_REGISTERED_AS": "Subject Unique Material Identifier",
    "WEIGHS": "The Subject's weight",
    "WAS_CREATED_IN": "The date of Subject creation",
    "WAS_DISSOLVED_IN": "The date of the Subject dissolution",
    "IS_OF_SIZE": "The number of Subject members",
    "OPERATES_IN": "The location where the Subject operates",
    "DIED_IN": "The Subject is a victim (died) of the event",
    "HAS_CATEGORY": "The Subject's category",
    "HAS_FAMILY_RELATIONSHIP": "The Subject and the Object are part ot the same family",
    "HAS_GENDER_FEMALE": "The context allows us to infer that the person is a female",
    "HAS_GENDER_MALE": "The context allows us to infer that the person is a male",
    "IS_BORN_IN": "The Subject's place of birth",
    "IS_BORN_ON": "The Subject's date of birth",
    "IS_DEAD_ON": "The Subject's date of death",
    "RESIDES_IN": "The Subject's place of residence",
    "HAS_LATITUDE": "The Subject's latitude",
    "HAS_LONGITUDE": "The Subject's longitude"
}

SUBJECT_TYPES = {
    'DIED_IN': 'PERSONNE',
}

OBJECT_TYPES = {
    'IS_LOCATED_IN': PLACE_ET,
    'HAS_CONTROL_OVER': ACTOR_ET + MATERIAL_ET + PLACE_ET,
    'IS_IN_CONTACT_WITH': ACTOR_ET,
    'OPERATES_IN': PLACE_ET,
    'STARTED_IN': PLACE_ET,
    'IS_AT_ODDS_WITH': ACTOR_ET,
    'IS_PART_OF': ORGANISATION_ET,
    'START_DATE': TIME_ET,
    'GENDER_MALE': PERSON_ET,
    'HAS_CATEGORY': CATEGORY_ET,
    'END_DATE': TIME_ET,
    'HAS_CONSEQUENCE': EVENT_ET,
    'INITIATED': EVENT_ET,
    'IS_OF_SIZE': QUANTITY_ET,
    'GENDER_FEMALE': PERSON_ET,
    'IS_COOPERATING_WITH': ACTOR_ET,
    'HAS_FAMILY_RELATIONSHIP': PERSON_ET,
    'RESIDES_IN': PLACE_ET,
    'HAS_QUANTITY': QUANTITY_ET,
    'IS_OF_NATIONALITY': NATIONALITY_ET,
    'CREATED': list(set(ORGANISATION_ET + MATERIAL_ET) - set(['GROUP_OF_INDIVIDUALS'])),
    'HAS_COLOR': COLOR_ET,
    'DEATHS_NUMBER': QUANTITY_ET,
    'INJURED_NUMBER': QUANTITY_ET,
    'IS_DEAD_ON': TIME_ET,
    'IS_BORN_IN': PLACE_ET,
    'DIED_IN': EVENT_ET,
    'WEIGHS': WEIGHT_ET,
    'IS_REGISTERED_AS': MATERIAL_REFERENCE_ET,
    'IS_BORN_ON': TIME_ET,
    'HAS_FOR_LENGTH': LENGTH_ET,
    'WAS_CREATED_IN': TIME_ET,
    'HAS_FOR_WIDTH': WIDTH_ET,
    'WAS_DISSOLVED_IN': TIME_ET,
    'HAS_FOR_HEIGHT': HEIGHT_ET,
    'HAS_LONGITUDE': LONGITUDE_ET,
    'HAS_LATITUDE': LATITUDE_ET,
}

ONE_CLASS = (CONFIG['one_class']['one_class_on'] == 'Yes')
print(f"One-vs-all mode? --> {ONE_CLASS}", flush=True)
def initialize_classes(one_class_name):
    # One-vs-all.
    if ONE_CLASS:
        # Load one-class mode.
        relation_ids = {
            f'NON-{one_class_name}': 0,
            one_class_name: 1,
        }
        CLASSES = list(relation_ids.keys())
        CLASS2ID = {class_: id_ for id_, class_ in enumerate(CLASSES)}
        ID2CLASS = {id_: class_ for class_, id_ in CLASS2ID.items()}
    else:
        # Load all classes mode.
        CLASSES = ["EMPTY"] + list(RELATIONS_IDS.keys())
        CLASS2ID = {class_: id_ for id_, class_ in enumerate(CLASSES)}
        ID2CLASS = {id_: class_ for class_, id_ in CLASS2ID.items()}
    return CLASSES, CLASS2ID, ID2CLASS