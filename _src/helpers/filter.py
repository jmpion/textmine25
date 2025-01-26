# ET stands for Entity Types.
ORGANISATION_ET = [
    'MILITARY_ORGANISATION',
    'NON_MILITARY_GOVERNMENT_ORGANISATION',
    'GROUP_OF_INDIVIDUALS',
    'INTERGOVERNMENTAL_ORGANISATION',
    'NON_GOVERNMENTAL_ORGANISATION',
]
PERSON_ET = [
    'CIVILIAN',
    'TERRORIST_OR_CRIMINAL',
    'MILITARY',
]
ACTOR_ET = ORGANISATION_ET + PERSON_ET
EVENT_ET = [
    'ACCIDENT',
    'CBRN_EVENT',
    'AGITATING_TROUBLE_MAKING',
    'CIVIL_WAR_OUTBREAK',
    'COUP_D_ETAT',
    'DEMONSTRATION',
    'ELECTION',
    'GATHERING',
    'ILLEGAL_CIVIL_DEMONSTRATION',
    'NATURAL_CAUSES_DEATH',
    'RIOT',
    'STRIKE',
    'SUICIDE',
    'BOMBING',
    'CRIMINAL_ARREST',
    'DRUG_OPERATION',
    'HOOLIGANISM_TROUBLEMAKING',
    'POLITICAL_VIOLENCE',
    'THEFT',
    'TRAFFICKING',
    'ECONOMICAL_CRISIS',
    'EPIDEMIC',
    'FIRE',
    'NATURAL_EVENT',
    'POLLUTION',
]
MATERIAL_ET = [
    'MATERIEL',
]
PLACE_ET = [
    'PLACE',
]
CATEGORY_ET = [
    'CATEGORY',
]
COLOR_ET = [
    'COLOR',
]
HEIGHT_ET = [
    'HEIGHT',
]
LENGTH_ET = [
    'LENGTH',
]
MATERIAL_REFERENCE_ET = [
    'MATERIAL_REFERENCE',
]
NATIONALITY_ET = [
    'NATIONALITY',
]
QUANTITY_ET = [
    'QUANTITY_EXACT',
    'QUANTITY_FUZZY',
    'QUANTITY_MAX',
    'QUANTITY_MIN',
]
TIME_ET = [
    'TIME_EXACT',
    'TIME_FUZZY',
    'TIME_MAX',
    'TIME_MIN',
]
WIDTH_ET = [
    'WIDTH',
]
WEIGHT_ET = [
    'WEIGHT',
]
LATITUDE_ET = [
    'LATITUDE',
]
LONGITUDE_ET = [
    'LONGITUDE',
]

ALL_ET = ACTOR_ET + EVENT_ET + MATERIAL_ET + PLACE_ET + NATIONALITY_ET + QUANTITY_ET + TIME_ET + COLOR_ET + HEIGHT_ET + LENGTH_ET + WIDTH_ET + MATERIAL_REFERENCE_ET + WEIGHT_ET + CATEGORY_ET + LATITUDE_ET + LONGITUDE_ET

CREATED_O_ET = list(set(ORGANISATION_ET + MATERIAL_ET) - set(['GROUP_OF_INDIVIDUALS']))
WAS_DISSOLVED_IN_S_ET = list(set(ORGANISATION_ET) - set(['GROUP_OF_INDIVIDUALS']))
WAS_CREATED_IN_S_ET = list(set(ORGANISATION_ET) - set(['GROUP_OF_INDIVIDUALS'])) + MATERIAL_ET
OPERATES_IN_S_ET = list(set(ORGANISATION_ET) - set(['GROUP_OF_INDIVIDUALS']))

def validate_relation(es_type, r_type, eo_type):
    """Returns True if the relation is authorized, False otherwise."""
    if r_type == 'IS_LOCATED_IN':
        return (es_type in (ACTOR_ET + EVENT_ET + PLACE_ET)) and eo_type in PLACE_ET
    if r_type == 'IS_OF_NATIONALITY':
        return (es_type in (ACTOR_ET + PLACE_ET)) and eo_type in NATIONALITY_ET
    if r_type == 'CREATED':
        return (es_type, eo_type) in [
            ('CIVILIAN', 'MATERIEL'),
            ('NON_GOVERNMENTAL_ORGANISATION', 'MATERIEL'), 
            ('GROUP_OF_INDIVIDUALS', 'MATERIEL'), 
            ('CIVILIAN', 'NON_GOVERNMENTAL_ORGANISATION'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'MATERIEL'),
            ('TERRORIST_OR_CRIMINAL', 'MATERIEL'),
            ('GROUP_OF_INDIVIDUALS', 'NON_GOVERNMENTAL_ORGANISATION'),
            ('MILITARY', 'MATERIEL'),
            ('NON_GOVERNEMENTAL_ORGANISATION', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
        ]
            # (es_type in ACTOR_ET) and (eo_type in (CREATED_O_ET)) # MATERIAL_ET condition added, but initial documentation did not include it.
    if r_type == 'HAS_CONTROL_OVER':
        return (es_type in ACTOR_ET) and (eo_type in (ACTOR_ET + MATERIAL_ET + PLACE_ET))
    if r_type == 'INITIATED':
        return (es_type, eo_type) in [
        ('CIVILIAN', 'ACCIDENT'),
        ('CIVILIAN', 'GATHERING'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'GATHERING'),
        ('GROUP_OF_INDIVIDUALS', 'GATHERING'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'CRIMINAL_ARREST'),
        ('GROUP_OF_INDIVIDUALS', 'DEMONSTRATION'),
        ('GROUP_OF_INDIVIDUALS', 'THEFT'),
        ('GROUP_OF_INDIVIDUALS', 'RIOT'),
        ('GROUP_OF_INDIVIDUALS', 'AGITATING_TROUBLE_MAKING'),
        ('CIVILIAN', 'SUICIDE'),
        ('GROUP_OF_INDIVIDUALS', 'STRIKE'),
        ('GROUP_OF_INDIVIDUALS', 'BOMBING'),
        ('GROUP_OF_INDIVIDUALS', 'CIVIL_WAR_OUTBREAK'),
        ('GROUP_OF_INDIVIDUALS', 'COUP_D_ETAT'),
        ('CIVILIAN', 'CBRN_EVENT'),
        ('GROUP_OF_INDIVIDUALS', 'POLITICAL_VIOLENCE'),
        ('CIVILIAN', 'DEMONSTRATION'),
        ('GROUP_OF_INDIVIDUALS', 'ILLEGAL_CIVIL_DEMONSTRATION'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'POLLUTION'),
        ('CIVILIAN', 'STRIKE'),
        ('GROUP_OF_INDIVIDUALS', 'FIRE'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'GATHERING'),
        ('TERRORIST_OR_CRIMINAL', 'TRAFFICKING'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'STRIKE'),
        ('TERRORIST_OR_CRIMINAL', 'COUP_D_ETAT'),
        ('GROUP_OF_INDIVIDUALS', 'DRUG_OPERATION'),
        ('TERRORIST_OR_CRIMINAL', 'THEFT'),
        ('CIVILIAN', 'AGITATING_TROUBLE_MAKING'),
        ('CIVILIAN', 'ILLEGAL_CIVIL_DEMONSTRATION'),
        ('GROUP_OF_INDIVIDUALS', 'TRAFFICKING'),
        ('GROUP_OF_INDIVIDUALS', 'ACCIDENT'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'CBRN_EVENT'),
        ('CIVILIAN', 'RIOT'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'DEMONSTRATION'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'COUP_D_ETAT'),
        ('CIVILIAN', 'POLLUTION'),
        ('CIVILIAN', 'FIRE'),
        ('GROUP_OF_INDIVIDUALS', 'CRIMINAL_ARREST'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'AGITATING_TROUBLE_MAKING'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'DRUG_OPERATION'),
        ('MILITARY_ORGANISATION', 'BOMBING'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'FIRE'),
        ('TERRORIST_OR_CRIMINAL', 'DRUG_OPERATION'),
        ('TERRORIST_OR_CRIMINAL', 'FIRE'),
        ('CIVILIAN', 'THEFT'),
        ('MILITARY_ORGANISATION', 'CRIMINAL_ARREST'),
        ('MILITARY_ORGANISATION', 'GATHERING'),
        ('TERRORIST_OR_CRIMINAL', 'BOMBING'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'BOMBING'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
        ('CIVILIAN', 'EPIDEMIC'),
        ('TERRORIST_OR_CRIMINAL', 'ACCIDENT'),
        ('GROUP_OF_INDIVIDUALS', 'POLLUTION'),
        ('GROUP_OF_INDIVIDUALS', 'HOOLIGANISM_TROUBLEMAKING'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'ECONOMICAL_CRISIS'),
        ('GROUP_OF_INDIVIDUALS', 'CBRN_EVENT'),
        ('TERRORIST_OR_CRIMINAL', 'SUICIDE'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'THEFT'),
        ('MILITARY_ORGANISATION', 'POLITICAL_VIOLENCE'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'CIVIL_WAR_OUTBREAK'),
        ('CIVILIAN', 'CIVIL_WAR_OUTBREAK'),
        ('TERRORIST_OR_CRIMINAL', 'HOOLIGANISM_TROUBLEMAKING'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'TRAFFICKING'),
        ('CIVILIAN', 'TRAFFICKING'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'DEMONSTRATION'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'RIOT'),
        ('NON_GOVERNMENTAL_ORGANISATION', 'ILLEGAL_CIVIL_DEMONSTRATION'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'POLITICAL_VIOLENCE'),
        ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'AGITATING_TROUBLE_MAKING'),
        ('MILITARY', 'CRIMINAL_ARREST'),
        ('MILITARY', 'COUP_D_ETAT')
    ]
        # return (es_type in ACTOR_ET) and (eo_type in EVENT_ET)
    if r_type == 'IS_AT_ODDS_WITH':
        return (es_type in ACTOR_ET) and (eo_type in ACTOR_ET)
    if r_type == 'IS_COOPERATING_WITH':
        return (es_type, eo_type) in [
            ('CIVILIAN', 'GROUP_OF_INDIVIDUALS'),
            ('GROUP_OF_INDIVIDUALS', 'CIVILIAN'),
            ('GROUP_OF_INDIVIDUALS', 'GROUP_OF_INDIVIDUALS'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'CIVILIAN'),
            ('CIVILIAN', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('NON_GOVERNMENTAL_ORGANISATION', 'NON_GOVERNMENTAL_ORGANISATION'),
            ('CIVILIAN', 'CIVILIAN'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'GROUP_OF_INDIVIDUALS'),
            ('GROUP_OF_INDIVIDUALS', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('GROUP_OF_INDIVIDUALS', 'NON_GOVERNMENTAL_ORGANISATION'),
            ('NON_GOVERNMENTAL_ORGANISATION', 'GROUP_OF_INDIVIDUALS'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('NON_GOVERNMENTAL_ORGANISATION', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'NON_GOVERNMENTAL_ORGANISATION'),
            ('NON_GOVERNMENTAL_ORGANISATION', 'CIVILIAN'),
            ('MILITARY_ORGANISATION', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'MILITARY_ORGANISATION'),
            ('CIVILIAN', 'NON_GOVERNMENTAL_ORGANISATION'),
            ('TERRORIST_OR_CRIMINAL', 'GROUP_OF_INDIVIDUALS'),
            ('GROUP_OF_INDIVIDUALS', 'TERRORIST_OR_CRIMINAL'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'INTERGOVERNMENTAL_ORGANISATION'),
            ('INTERGOVERNMENTAL_ORGANISATION', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'TERRORIST_OR_CRIMINAL'),
            ('TERRORIST_OR_CRIMINAL', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('MILITARY_ORGANISATION', 'CIVILIAN'),
            ('CIVILIAN', 'MILITARY_ORGANISATION'),
            ('MILITARY', 'NON_MILITARY_GOVERNMENT_ORGANISATION'),
            ('MILITARY', 'CIVILIAN'),
            ('CIVILIAN', 'MILITARY'),
            ('GROUP_OF_INDIVIDUALS', 'INTERGOVERNMENTAL_ORGANISATION'),
            ('INTERGOVERNMENTAL_ORGANISATION', 'GROUP_OF_INDIVIDUALS'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'MILITARY')
        ]
        # return (es_type in ACTOR_ET) and (eo_type in ACTOR_ET)
    if r_type == 'IS_IN_CONTACT_WITH':
        return (es_type in ACTOR_ET) and (eo_type in ACTOR_ET)
    if r_type == 'IS_PART_OF':
        return (es_type in ACTOR_ET) and (eo_type in ORGANISATION_ET)
    if r_type == 'DEATHS_NUMBER':
        return (es_type in EVENT_ET) and (eo_type in QUANTITY_ET)
    if r_type == 'END_DATE':
        return (es_type in EVENT_ET) and (eo_type in TIME_ET)
    if r_type == 'HAS_CONSEQUENCE':
        return (es_type in EVENT_ET) and (eo_type in EVENT_ET)
    if r_type == 'INJURED_NUMBER':
        return (es_type in EVENT_ET) and (eo_type in QUANTITY_ET)
    if r_type == 'START_DATE':
        return (es_type in EVENT_ET) and (eo_type in TIME_ET)
    if r_type == 'STARTED_IN':
        return (es_type in EVENT_ET) and (eo_type in PLACE_ET)
    if r_type == 'HAS_COLOR':
        return (es_type in MATERIAL_ET) and (eo_type in COLOR_ET)
    if r_type == 'HAS_FOR_HEIGHT':
        return (es_type in MATERIAL_ET) and (eo_type in HEIGHT_ET)
    if r_type == 'HAS_FOR_LENGTH':
        return (es_type in MATERIAL_ET) and (eo_type in LENGTH_ET)
    if r_type == 'HAS_FOR_WIDTH':
        return (es_type in MATERIAL_ET) and (eo_type in WIDTH_ET)
    if r_type == 'HAS_QUANTITY':
        return (es_type in MATERIAL_ET) and (eo_type in QUANTITY_ET)
    if r_type == 'IS_REGISTERED_AS':
        return (es_type in MATERIAL_ET) and (eo_type in MATERIAL_REFERENCE_ET)
    if r_type == 'WEIGHS':
        return (es_type in MATERIAL_ET) and (eo_type in WEIGHT_ET)
    if r_type == 'WAS_CREATED_IN':
        return (es_type, eo_type) in [
            ('MATERIEL', 'TIME_EXACT'),
            ('NON_GOVERNMENTAL_ORGANISATION', 'TIME_EXACT'),
            ('NON_MILITARY_GOVERNMENT_ORGANISATION', 'TIME_EXACT'),
            ('MATERIEL', 'TIME_FUZZY'),
            ('MATERIEL', 'TIME_MIN')
        ]
        # return (es_type in WAS_CREATED_IN_S_ET) and (eo_type in TIME_ET) # MATERIAL_ET added for subject type, but the initial documentation did not include it.
    if r_type == 'WAS_DISSOLVED_IN':
        return (es_type in WAS_DISSOLVED_IN_S_ET) and (eo_type in TIME_ET)
    if r_type == 'IS_OF_SIZE':
        return (es_type in ORGANISATION_ET) and (eo_type in QUANTITY_ET)
    if r_type == 'OPERATES_IN':
        return (es_type in OPERATES_IN_S_ET) and (eo_type in PLACE_ET)
    if r_type == 'DIED_IN':
        return (es_type in PERSON_ET) and (eo_type in EVENT_ET)
    if r_type == 'HAS_CATEGORY':
        return (es_type in PERSON_ET) and (eo_type in CATEGORY_ET)
    if r_type == 'HAS_FAMILY_RELATIONSHIP':
        return (es_type in PERSON_ET) and (eo_type in PERSON_ET)
    if r_type == 'GENDER_FEMALE':
        return (es_type in PERSON_ET) and (eo_type in PERSON_ET)
    if r_type == 'GENDER_MALE':
        return (es_type in PERSON_ET) and (eo_type in PERSON_ET)
    if r_type == 'IS_BORN_IN':
        return (es_type in PERSON_ET) and (eo_type in PLACE_ET)
    if r_type == 'IS_BORN_ON':
        return (es_type in PERSON_ET) and (eo_type in TIME_ET)
    if r_type == 'IS_DEAD_ON':
        return (es_type in PERSON_ET) and (eo_type in TIME_ET)
    if r_type == 'RESIDES_IN':
        return (es_type in PERSON_ET) and (eo_type in PLACE_ET)
    if r_type == 'HAS_LATITUDE':
        return (es_type in PLACE_ET) and (eo_type in LATITUDE_ET)
    if r_type == 'HAS_LONGITUDE':
        return (es_type in PLACE_ET) and (eo_type in LONGITUDE_ET)
    return True

def validate_pair(es_type, eo_type):
    """Takes the type of a pair of entity subject and entity object and outputs whether there could exist a
    non-empty relation (True), or not (False)."""
    if (es_type in ACTOR_ET) and (eo_type in (PLACE_ET + NATIONALITY_ET + ORGANISATION_ET + ACTOR_ET + MATERIAL_ET + EVENT_ET)):
        return True
    if (es_type in EVENT_ET) and (eo_type in (QUANTITY_ET + TIME_ET + EVENT_ET + PLACE_ET)):
        return True
    if (es_type in MATERIAL_ET) and (eo_type in (COLOR_ET + HEIGHT_ET + LENGTH_ET + WIDTH_ET + QUANTITY_ET + MATERIAL_REFERENCE_ET + WEIGHT_ET + TIME_ET)):
        return True
    if (es_type in ORGANISATION_ET) and (eo_type in (TIME_ET + QUANTITY_ET + PLACE_ET)):
        return True
    if (es_type in PERSON_ET) and (eo_type in (EVENT_ET + CATEGORY_ET + PERSON_ET + PLACE_ET + TIME_ET)):
        return True
    if (es_type in PLACE_ET) and (eo_type in (PLACE_ET + NATIONALITY_ET + LATITUDE_ET + LONGITUDE_ET)):
        return True
    return False