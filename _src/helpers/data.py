from datasets import Dataset as hf_Dataset
from datasets import load_dataset
import json
import pandas as pd
import random

from .constants import CONFIG
from .entity_markers import add_entity_markers, special_tokens_from_wrapper_mode
from .filter import validate_pair, validate_relation

DATA_PATH = "data"

def load_train_data(data_path):
    """
    Load the train data from the specified data path.
    The only data transformation consists of transforming the entities and relations from strings to lists.

    Parameters:
    data_path (str): The path to the train data.

    Returns:
    pd.DataFrame: The loaded train data with entities and relations parsed.
    """
    train_df = pd.read_csv(data_path + "/train.csv") # chargement des données d'entraînement
    train_df = train_df.set_index("id")
    train_df.entities = train_df.entities.apply(json.loads) # parsing des entités
    train_df.relations = train_df.relations.apply(json.loads) # parsing des relations
    return train_df

def load_test_data(data_path):
    """
    Load the test data from the specified data path.
    The only data transformation consists of transforming the entities and relations from strings to lists.
    
    Parameters:
    data_path (str): The path to the test data.
    
    Returns:
    pd.DataFrame: The loaded test data with entities parsed.
    """
    test_df = pd.read_csv(data_path + "/test_01-07-2024.csv") # chargement des données d'entraînement
    test_df = test_df.set_index("id")
    test_df.entities = test_df.entities.apply(json.loads) # parsing des entités
    return test_df

def prepare_dataset(data_files):
    """
    Prepare a dataset from the specified data files.
    Three things are done:
    - the dataset is loaded from its path.
    - the entities are parsed.
    - the relations are parsed.

    Parameters:
    data_files (str): The path to the CSV file to load.

    Returns:
    datasets.Dataset: The loaded dataset with entities and relations parsed.
    Attributes:
    - id (int): The id of the data.
    - text (str): The text of the data.
    - entities (list[dict[str, int | list[dict[str, str | int]] | int]]): The list of parsed entities.
    - relations (list[list[int | str]]): The list of parsed relations.
    """
    dataset = load_dataset('csv', data_files=data_files)
    dataset = dataset.map(convert_string_to_dict_entities)
    dataset = dataset.map(convert_string_to_dict_relations)
    return dataset

def convert_string_to_dict_entities(example):
    example['entities'] = json.loads(example['entities'])  # Convert the string to a dictionary
    return example

def convert_string_to_dict_relations(example):
    tmp_relations = json.loads(example['relations'])  # Convert the string to a dictionary
    example['relations'] = [[str(_) for _ in __] for __ in tmp_relations] # Converts all elements of the triplets to strings.
    # ... necessary for using datasets Dataset object.
    return example

# Function to create the new dataset
def extract_entity_pairs_and_relations(original_dataset: hf_Dataset, one_class_name: str='') -> hf_Dataset:
    """Converts a dataset from the default format to a dataset having entity pairs to which all of their relations are associated.
    The current version extracts all entity pairs and does not filter out wrong types of entity pairs.

    Args:
    - original_dataset (datasets.Dataset): original dataset.
        Attributes:
        - id (int): The report id.
        - text (str): The text.
        - entities (list[dict[str, int | list[dict[str, str | int]] | int]]): The list of entities.
        - relations (list[list[str]]): The list of relations, which includes empty relations.

    Returns:
    - datasets.Dataset: new dataset.
        Attributes:
        - id (int): The report id.
        - text (str): The text, unchanged.
        - entity_pair (list[dict[str, int | list[dict[str, str | int]] | int]]): The entity pair.
        - relation_types (list[str]): The list of relations for this pair of entities.
    """
    ids = []
    texts = []
    entity_pairs = []
    relation_types_list = []

    SUPPORT = CONFIG['one_class']['support']
    ONE_CLASS = CONFIG['one_class']['one_class_on']
    if SUPPORT == 'Yes':
        assert ONE_CLASS == 'Yes'
        assert one_class_name != '', 'Please provide the name of the one-class class.'
        RELATION_TYPE = one_class_name

    # Iterate through the original dataset.
    for example in original_dataset:
        text = example['text']
        entities = example['entities']
        relations = example['relations']

        # Create a mapping from entity id to the whole entity object.
        entity_map = {entity['id']: entity for entity in entities}

        # Dictionary to group relation types by entity pair
        relation_dict = {}

        # Iterate through relations
        for subject_id, relation_type, object_id in relations:
            subject_id, object_id = int(subject_id), int(object_id)
            entity_pair = (subject_id, object_id)

            if (SUPPORT=='No') or validate_relation(entity_map[subject_id]['type'], RELATION_TYPE, entity_map[object_id]['type']):
                # Add relation types to the entity pair
                if entity_pair not in relation_dict:
                    relation_dict[entity_pair] = []
                relation_dict[entity_pair].append(relation_type)

        # Add the data for this example to the new dataset
        for (subject_id, object_id), relation_types in relation_dict.items():
            subject_id, object_id = int(subject_id), int(object_id)
            entity_pair = (entity_map[subject_id], entity_map[object_id])  # Convert entity ids to actual entity mentions
            ids.append(example['id'])
            texts.append(text)
            entity_pairs.append(entity_pair)
            relation_types_list.append(relation_types)

    # Create the new dataset
    new_data = {
        'id': ids,
        'text': texts,
        'entity_pair': entity_pairs,
        'relation_types': relation_types_list
    }

    return hf_Dataset.from_dict(new_data)

# Function to create the new dataset
def extract_entity_pairs(original_dataset: hf_Dataset, one_class_name: str='') -> hf_Dataset:
    texts = []
    subject_ids = []
    object_ids = []
    entity_pairs = []
    ids = []

    # Iterate through the dataset
    for example in original_dataset:
        text = example['text']
        entities = example['entities']
        for es in entities:
            for eo in entities:
                if one_class_name == '':
                    if validate_pair(es['type'], eo['type']):
                        texts.append(text)
                        subject_ids.append(es['id'])
                        object_ids.append(eo['id'])
                        entity_pairs.append([es, eo])
                        ids.append(example['id'])
                else:
                    if validate_relation(es['type'], one_class_name, eo['type']):
                        texts.append(text)
                        subject_ids.append(es['id'])
                        object_ids.append(eo['id'])
                        entity_pairs.append([es, eo])
                        ids.append(example['id'])
    # Create the new dataset
    new_data = {
        'id': ids,
        'text': texts,
        'subject_id': subject_ids,
        'object_id': object_ids,
        'entity_pair': entity_pairs,
    }

    return hf_Dataset.from_dict(new_data)

def load_inject_empty_relations(p_empty: float):
    """Load a function for preprocessing a dataset, to inject empty relations.

    Args:
        p_empty (float): probability threshold for assigning empty label.

    Returns:    
        inject_empty_relations: function for injecting empty relations.
    """
    assert 0. <= p_empty <= 1.
    def inject_empty_relations(example):
        """
        Arguments:
        - example (dict): A dictionary containing the data. 
            Attributes:
            - id (int): The id of the data.
            - text (str): The text of the data.
            - entities (list[dict[str, int | list[dict[str, str | int]] | int]]): The list of parsed entities.
            - relations (list[list[str]]): The list of parsed relations.
        
        Returns:
        - example (dict): A dictionary containing the data, with added empty relations.
            Attributes:
            - id (int): The id of the data.
            - text (str): The text of the data.
            - entities (list[dict[str, int | list[dict[str, str | int]] | int]]): The list of parsed entities.
            - relations (list[list[str]]): The list of parsed relations, including empty relations.
        """
        Ei = example['entities']
        Ri = example['relations']

        relation_dict = {}
        # Fill in the existing relations.
        for r in Ri:
            # Unpack relation and ensure the write types.
            es_id, rt, eo_id = r[0], r[1], r[2]
            es_id = int(es_id)
            eo_id = int(eo_id)

            assert isinstance(es_id, int)
            assert isinstance(eo_id, int)
            
            if es_id not in relation_dict:
                relation_dict[es_id] = {}
            if eo_id not in relation_dict[es_id]:
                relation_dict[es_id][eo_id] = [rt]
            else:
                relation_dict[es_id][eo_id].append(rt)
        # Then, check for all entity pairs if there is a relation.
        for es in Ei:
            es_id = es['id']
            assert isinstance(es_id, int)
            for eo in Ei:
                eo_id = eo['id']
                assert isinstance(eo_id, int)

                # Picking empty relations.
                if (es_id not in relation_dict) or (eo_id not in relation_dict[es_id]):
                    if validate_pair(es['type'], eo['type']):
                        p_select = random.random()
                        if p_select <= p_empty:
                            example['relations'].append([str(es_id), 'EMPTY', str(eo_id)])
        return example
    return inject_empty_relations

def extract_token_indices(entity, offset_mapping_, input_ids, extraction_mode='all tokens', special_token=None, tokenizer=None):
    """
    Extracts token indices for the given entity in the given text.

    Args:
        entity: the entity for which to extract the token indices.
        offset_mapping_: the offset mapping of the input text.
        extraction_mode (optional): the mode for extracting the token indices. Either 'all tokens' to extract indices of all normal tokens contained in the entity mentions.
        Or 'special_tokens' to extract only the indices of the special tokens in the entity mentions. The starting entity markers in fact.
        special_token (str): the special token for which to extract the token indices, in the case where extraction_mode=='special_tokens'.
    Returns:
        a list of token indices for the given entity.
    """
    if extraction_mode=='all_tokens':
        for mention in entity['mentions']:
            span_start_char = mention['start']
            span_end_char = mention['end']
            token_indices = []
            for idx, (start, end) in enumerate(offset_mapping_):
                if span_start_char < end <= span_end_char or span_start_char <= start < span_end_char:
                    token_indices.append(idx)
    elif extraction_mode=='special_tokens':
        # Get the ID of the special token
        special_token_id = tokenizer.convert_tokens_to_ids(special_token)
        token_indices = [i for i, token_id in enumerate(input_ids) if token_id==special_token_id]
    return token_indices

def get_relation_token_indices_bounds(entity_subject, entity_object, insert_positions):
    min_start = 0
    min_set = False
    for mention in entity_subject['mentions'] + entity_object['mentions']:
        start_tmp = mention['start']
        for insert_position in insert_positions:
            if insert_position <= start_tmp:
                start_tmp += len(insert_positions[insert_position])
        if not min_set:
            min_set = True
            min_start = start_tmp
        if start_tmp < min_start:
            min_start = start_tmp

    max_end = 0
    max_set = False
    for mention in entity_subject['mentions'] + entity_object['mentions']:
        end_tmp = mention['end']
        for insert_position in insert_positions:
            if insert_position <= end_tmp:
                end_tmp += len(insert_positions[insert_position])
        if not max_set:
            max_set = True
            max_end = end_tmp
        if end_tmp > max_end:
            max_end = end_tmp
    return min_start, max_end

def load_preprocess_function(tokenizer, classes, class2id, one_class_name: str=''):
    # Prépare une fonction de pré-traitement qui  réalise les étapes suivantes :
    # 1. Marquage des entités.
    # 2. Tokenization.
    # 3. Récupération des indices des tokens des entités.
    # 4. Préparation des vecteurs de labels. 
    def preprocess_function(example):
        # 1. Add entity markers for the entity pair, if required.
        ENTITY_MARKERS_ON = CONFIG['entity_markers']['entity_markers_on']
        if ENTITY_MARKERS_ON == 'Yes':
            text, insert_positions = add_entity_markers(example['text'], example['entity_pair'])
        else:
            text = example['text']

        # 2. Tokenize the text.
        tokens = tokenizer.encode_plus(text, return_offsets_mapping=True)
        example['text'] = text
        example['input_ids'] = tokens['input_ids']
        example['attention_mask'] = tokens['attention_mask']
        example['offset_mapping'] = tokens['offset_mapping']

        # 3. Get token ids for entity pair mentions, if required.
        SEPARATE_SPANS_EMBEDDINGS = CONFIG['separate_spans_embeddings']
        if SEPARATE_SPANS_EMBEDDINGS=='Yes':
            entity_pair = example['entity_pair']
            entity_subject = entity_pair[0]
            entity_object = entity_pair[1]
            tokens = example['input_ids']
            offset_mapping = example['offset_mapping']

            # Get special tokens, depending on the wrapper mode and the entities types.
            WRAPPER_MODE = CONFIG['entity_markers']['wrapper_mode']
            special_tokens = special_tokens_from_wrapper_mode(WRAPPER_MODE, entity_subject['type'], entity_object['type'])
            
            special_token_subject_start = special_tokens['special_token_subject_start']
            special_token_object_start = special_tokens['special_token_object_start']
            special_token_subject_end = special_tokens['special_token_subject_end']
            special_token_object_end = special_tokens['special_token_object_end']

            # Extract token indices.
            extraction_mode = CONFIG['extraction_mode']
            example['subject_token_ids'] = extract_token_indices(entity_subject, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_subject_start, tokenizer=tokenizer)
            example['object_token_ids'] = extract_token_indices(entity_object, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_object_start, tokenizer=tokenizer)
            
            example['subject_token_ids_end'] = extract_token_indices(entity_subject, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_subject_end, tokenizer=tokenizer)
            example['object_token_ids_end'] = extract_token_indices(entity_object, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_object_end, tokenizer=tokenizer)
            
            # Ensure that subject_token_ids and object_token_ids have a fixed length
            max_token_span_length = 20
            example['subject_token_ids'] = example['subject_token_ids'] + \
                                           [0] * max(0, max_token_span_length - len(example['subject_token_ids']))
            example['object_token_ids'] = example['object_token_ids'] + \
                                           [0] * max(0, (max_token_span_length - len(example['object_token_ids'])))

            example['subject_token_ids_end'] = example['subject_token_ids_end'] + \
                                           [0] * max(0, max_token_span_length - len(example['subject_token_ids_end']))
            example['object_token_ids_end'] = example['object_token_ids_end'] + \
                                           [0] * max(0, (max_token_span_length - len(example['object_token_ids_end'])))
            
            min_start, max_end = get_relation_token_indices_bounds(entity_subject, entity_object, insert_positions)
                    
            span_start_char = min_start
            span_end_char = max_end
            relation_token_indices = []
            for idx, (start, end) in enumerate(offset_mapping):
                if span_start_char < end <= span_end_char or span_start_char <= start < span_end_char:
                    relation_token_indices.append(idx)
            example['relation_token_ids'] = relation_token_indices            
            example['relation_token_ids'] = example['relation_token_ids'] + \
                                           [0] * max(0, (512 - len(example['relation_token_ids'])))
            
        # Prepare the label binary vector.
        all_labels = example['relation_types']
        labels = [0. for _ in range(len(classes))]
        ONE_CLASS = (CONFIG['one_class']['one_class_on']=='Yes')
        if not ONE_CLASS:
            for label in all_labels:
                label_id = class2id[label]
                labels[label_id] = 1.
        else:
            ONE_CLASS_NAME = one_class_name
            assert one_class_name != '', "Please provide the name of the one-class class."
            if ONE_CLASS_NAME in all_labels:
                labels = [0., 1.]
            else:
                labels = [1., 0.]
        
        # Specify the labels.
        example['labels'] = labels
        return example
    return preprocess_function

def load_preprocess_function_for_inference(tokenizer):
    def preprocess_function(example):
        # 1. Add entity markers for the entity pair, if required.
        ENTITY_MARKERS_ON = CONFIG['entity_markers']['entity_markers_on']
        if ENTITY_MARKERS_ON == 'Yes':
            text, insert_positions = add_entity_markers(example['text'], example['entity_pair'])
        else:
            text = example['text']

        # 2. Tokenize the text.
        tokens = tokenizer.encode_plus(text, return_offsets_mapping=True)
        example['text'] = text
        example['input_ids'] = tokens['input_ids']
        example['attention_mask'] = tokens['attention_mask']
        example['offset_mapping'] = tokens['offset_mapping']

        # 3. Get token ids for entity pair mentions, if required.
        SEPARATE_SPANS_EMBEDDINGS = CONFIG['separate_spans_embeddings']
        if SEPARATE_SPANS_EMBEDDINGS=='Yes':
            entity_pair = example['entity_pair']
            entity_subject = entity_pair[0]
            entity_object = entity_pair[1]
            tokens = example['input_ids']
            offset_mapping = example['offset_mapping']
            
            extraction_mode = CONFIG['extraction_mode']
            WRAPPER_MODE = CONFIG['entity_markers']['wrapper_mode']
            special_tokens = special_tokens_from_wrapper_mode(WRAPPER_MODE, entity_subject['type'], entity_object['type'])
            
            special_token_subject_start = special_tokens['special_token_subject_start']
            special_token_object_start = special_tokens['special_token_object_start']
            special_token_subject_end = special_tokens['special_token_subject_end']
            special_token_object_end = special_tokens['special_token_object_end']

            example['subject_token_ids'] = extract_token_indices(entity_subject, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_subject_start, tokenizer=tokenizer)
            example['object_token_ids'] = extract_token_indices(entity_object, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_object_start, tokenizer=tokenizer)
            
            example['subject_token_ids_end'] = extract_token_indices(entity_subject, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_subject_end, tokenizer=tokenizer)
            example['object_token_ids_end'] = extract_token_indices(entity_object, offset_mapping, example['input_ids'], extraction_mode=extraction_mode, special_token=special_token_object_end, tokenizer=tokenizer)
            
            # Ensure that subject_token_ids and object_token_ids have a fixed length
            max_token_span_length = 20
            example['subject_token_ids'] = example['subject_token_ids'] + \
                                           [0] * max(0, max_token_span_length - len(example['subject_token_ids']))
            example['object_token_ids'] = example['object_token_ids'] + \
                                           [0] * max(0, (max_token_span_length - len(example['object_token_ids'])))

            example['subject_token_ids_end'] = example['subject_token_ids_end'] + \
                                           [0] * max(0, max_token_span_length - len(example['subject_token_ids_end']))
            example['object_token_ids_end'] = example['object_token_ids_end'] + \
                                           [0] * max(0, (max_token_span_length - len(example['object_token_ids_end'])))
            
            min_start, max_end = get_relation_token_indices_bounds(entity_subject, entity_object, insert_positions)
                    
            span_start_char = min_start
            span_end_char = max_end
            relation_token_indices = []
            for idx, (start, end) in enumerate(offset_mapping):
                if span_start_char < end <= span_end_char or span_start_char <= start < span_end_char:
                    relation_token_indices.append(idx)
            example['relation_token_ids'] = relation_token_indices            
            example['relation_token_ids'] = example['relation_token_ids'] + \
                                           [0] * max(0, (512 - len(example['relation_token_ids'])))
        
        example['subject_id'] = int(example['entity_pair'][0]['id'])
        example['object_id'] = int(example['entity_pair'][1]['id'])
        example['id'] = int(example['id'])

        return example
    return preprocess_function

def prepare_relation_datasets(train_dataset, val_dataset, one_class_name):
    train_dataset = train_dataset.map(load_inject_empty_relations(1.))
    val_dataset = val_dataset.map(load_inject_empty_relations(1.))
    train_dataset_new = extract_entity_pairs_and_relations(train_dataset, one_class_name)
    val_dataset_new = extract_entity_pairs_and_relations(val_dataset, one_class_name)
    return train_dataset_new, val_dataset_new