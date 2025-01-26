import argparse
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from safetensors.torch import load_file
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from helpers.constants import CONFIG, initialize_classes
from helpers.data import convert_string_to_dict_entities, extract_entity_pairs, load_preprocess_function_for_inference
from helpers.entity_markers import additional_tokens_from_wrapper_mode
from helpers.model import load_model

def load_best_model_and_tokenizer(base_model_path, model_weights_path, num_classes, one_class_name):
    """
    Load the best model checkpoint using the trainer state file, and initialize the tokenizer.
    """
    print(one_class_name)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(num_classes, base_model_path, one_class_name=one_class_name).to(DEVICE)
    
    # Find the best checkpoint using the trainer state file
    with open(os.path.join(model_weights_path, os.listdir(model_weights_path)[-1], 'trainer_state.json'), 'r') as f:
        trainer_state = json.load(f)
    best_checkpoint = os.path.basename(trainer_state['best_model_checkpoint'])
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_weights_path, best_checkpoint), cache_dir='NON_EXISTENT_DIR')

    # Add special tokens if entity markers are enabled
    ENTITY_MARKERS_ON = CONFIG['entity_markers']['entity_markers_on']
    if ENTITY_MARKERS_ON == 'Yes':
        WRAPPER_MODE = CONFIG['entity_markers']['wrapper_mode']
        added_special_tokens = additional_tokens_from_wrapper_mode(WRAPPER_MODE)
        tokenizer.add_special_tokens({'additional_special_tokens': added_special_tokens})
        model.resize_token_embeddings(len(tokenizer) + 1)

    # Load the model state dict from safetensors
    state_dict = load_file(os.path.join(model_weights_path, best_checkpoint, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, tokenizer

def generate_predictions(model, tokenizer, dataset, initial_dataset_ids, ID2CLASS, DEVICE='cuda'):
    """
    Generate predictions using the loaded model and tokenizer on the tokenized dataset.
    """
    outputs_by_id = {}
    with torch.no_grad():
        for example in tqdm(dataset, desc="Generating predictions"):
            # Generate logits using the model
            logits = model(
                input_ids=torch.tensor(example['input_ids']).unsqueeze(0).to(DEVICE),
                attention_mask=torch.tensor(example['attention_mask']).unsqueeze(0).to(DEVICE),
                subject_token_ids=torch.tensor(example['subject_token_ids']).unsqueeze(0).to(DEVICE),
                object_token_ids=torch.tensor(example['object_token_ids']).unsqueeze(0).to(DEVICE),
                subject_token_ids_end=torch.tensor(example['subject_token_ids_end']).unsqueeze(0).to(DEVICE),
                object_token_ids_end=torch.tensor(example['object_token_ids_end']).unsqueeze(0).to(DEVICE),
                relation_token_ids=torch.tensor(example['relation_token_ids']).unsqueeze(0).to(DEVICE),
            )
            
            # Convert logits to binary predictions using a threshold of 0.5
            predictions = torch.sigmoid(logits)
            predictions = (predictions >= 0.5).squeeze().detach().cpu().numpy().astype(int)
            
            # Get the id of the current example
            id_ = example['id']
            if id_ not in outputs_by_id:
                outputs_by_id[id_] = []
            
            # Add relations if predicted
            for k in range(len(predictions)):
                if predictions[k] == 1 and k != 0:
                    outputs_by_id[id_].append([int(example['subject_id']), ID2CLASS[k], int(example['object_id'])])

    # Ensure all initial dataset IDs are covered
    for id_ in initial_dataset_ids:
        if id_ not in outputs_by_id:
            outputs_by_id[id_] = []
    
    return outputs_by_id

def main(output_dir, base_model_path, one_class_name=""):
    set_seed(0)

    # Load classes and initialize variables
    CLASSES, CLASS2ID, ID2CLASS = initialize_classes(one_class_name)
    NUM_CLASSES = len(CLASSES)

    # Determine submission type
    SUBMISSION_TYPE = CONFIG['submission_type']
    if SUBMISSION_TYPE == 'TEST':
        dataset = load_dataset('csv', data_files='data/test_01-07-2024.csv')
    elif SUBMISSION_TYPE == 'VALIDATION':
        dataset = load_dataset('csv', data_files='data/train.csv')
    else:
        raise Exception("Wrong submission_type in the config.yml")

    # Convert string entities to Python objects
    dataset = dataset.map(convert_string_to_dict_entities)
    dataset = dataset['train']
    initial_dataset_ids = dataset['id']

    # Extract entity pairs and preprocess
    dataset = extract_entity_pairs(dataset, one_class_name=one_class_name)
    model_paths = [f"{output_dir}/{one_class_name}_fold_{i + 1}" for i in range(5)]

    all_model_outputs = []

    for model_path in model_paths:
        # Load best model and tokenizer
        model, tokenizer = load_best_model_and_tokenizer(base_model_path, model_path, NUM_CLASSES, one_class_name)
        
        preprocess_function = load_preprocess_function_for_inference(tokenizer)
        tokenized_dataset = dataset.map(preprocess_function)

        # Generate predictions using the current model
        outputs_by_id = generate_predictions(model, tokenizer, tokenized_dataset, initial_dataset_ids, ID2CLASS)
        all_model_outputs.append(outputs_by_id)

    # Perform hard voting across models
    final_outputs_by_id = {}
    for id_ in initial_dataset_ids:
        relation_votes = {}
        for model_output in all_model_outputs:
            for relation in model_output.get(id_, []):
                relation_tuple = tuple(relation)
                if relation_tuple not in relation_votes:
                    relation_votes[relation_tuple] = 0
                relation_votes[relation_tuple] += 1
        
        final_outputs_by_id[id_] = [list(rel) for rel, count in relation_votes.items() if count >= 3]

    # Create a submission dataframe
    data_dict = {'id': list(final_outputs_by_id.keys()), 'relations': list(final_outputs_by_id.values())}
    df_submission = pd.DataFrame(data_dict)
    df_submission['relations'] = df_submission['relations'].apply(str)

    # Save the final submission
    df_submission.to_csv(f'_results/{one_class_name}-{SUBMISSION_TYPE}_ensemble_MODELTOWRITE.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory containing the trained models.')
    parser.add_argument('--base_model_path', type=str, required=True, help='Base path to the transformer model.')
    parser.add_argument('--class_names', nargs='+', help='Names of the one-class classes.', default="")
    args = parser.parse_args()

    for one_class_name in args.class_names:
        main(args.output_dir, args.base_model_path, one_class_name=one_class_name)
