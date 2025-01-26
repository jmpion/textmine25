# Idea: This script aims at generating a submission from the test csv
# ... given a trained model for relation extraction.
print("Getting started....", flush=True)

# Imports.
import argparse
from datasets import load_dataset
import json
import os
import pandas as pd
from safetensors.torch import load_file
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
# Custom imports.
from helpers.constants import CONFIG, initialize_classes
from helpers.data import convert_string_to_dict_entities, extract_entity_pairs, load_preprocess_function_for_inference
from helpers.entity_markers import additional_tokens_from_wrapper_mode
from helpers.model import load_model


def main(base_model_path, one_class_name=""):
    # Reproducibility.
    set_seed(0)

    # Load model.
    BASE_MODEL_PATH = CONFIG['model']['name']
    MODEL_PATH = f"{base_model_path}{one_class_name}"

    # If only one class.
    ONE_CLASS = CONFIG['one_class']['one_class_on']
    ONE_CLASS_NAME = one_class_name
    if ONE_CLASS=='Yes':
        assert ONE_CLASS_NAME != "", "Please provide the name of the one-class class."
        print(f"One class: {ONE_CLASS_NAME}", flush=True)
    CLASSES, CLASS2ID, ID2CLASS = initialize_classes(ONE_CLASS_NAME)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = len(CLASSES)
    model = load_model(NUM_CLASSES, BASE_MODEL_PATH, one_class_name=ONE_CLASS_NAME).to(DEVICE)

    # Looking for the best checkpoint:
    with open(os.path.join(MODEL_PATH, os.listdir(MODEL_PATH)[-1], 'trainer_state.json'), 'r') as f:
        trainer_state = json.load(f)
    best_checkpoint = os.path.basename(trainer_state['best_model_checkpoint'])

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, best_checkpoint), cache_dir='NON_EXISTENT_DIR')

    ENTITY_MARKERS_ON = CONFIG['entity_markers']['entity_markers_on']
    if ENTITY_MARKERS_ON=='Yes':
        # Add special tokens (entity markers) to the tokenizer.
        WRAPPER_MODE = CONFIG['entity_markers']['wrapper_mode']
        added_special_tokens = additional_tokens_from_wrapper_mode(WRAPPER_MODE)
        special_tokens = {
            'additional_special_tokens': added_special_tokens
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer) + 1)

    # Load the state dict from the safetensors file
    state_dict = load_file(os.path.join(MODEL_PATH, best_checkpoint, 'model.safetensors'))

    # Load state dict with strict=False to ignore unexpected keys
    model.load_state_dict(state_dict, strict=False)

    # Set model on eval mode.
    model.eval()

    # Load dataset.
    SUBMISSION_TYPE = CONFIG['submission_type']
    print(f"Submission type: {SUBMISSION_TYPE}")
    if SUBMISSION_TYPE=='TEST':
        dataset = load_dataset('csv', data_files='data/test_01-07-2024.csv')
    elif SUBMISSION_TYPE=='VALIDATION':
        dataset = load_dataset('csv', data_files='data/train.csv')
    else:
        raise Exception("Wrong submission_type in the config.yml")

    ## Preprocess dataset.
    # Convert strange string entities to Python objets.
    dataset = dataset.map(convert_string_to_dict_entities)

    # Access sub-dataset.
    dataset = dataset['train']

    # Initial dataset ids.
    initial_dataset_ids = dataset['id']

    # Create new dataset based on entity pairs.
    dataset = extract_entity_pairs(dataset, one_class_name=ONE_CLASS_NAME)

    # Tokenize the dataset texts.
    preprocess_function = load_preprocess_function_for_inference(tokenizer)
    tokenized_dataset = dataset.map(preprocess_function)

    assert len(tokenized_dataset) == len(dataset), print("Length of tokenized dataset:", len(tokenized_dataset), "\nLength of initial dataset:", len(dataset))

    # Fill in a dictionary containing the lists of relations for each example id in the test set.
    outputs_by_id = {}
    with torch.no_grad():
        for example in tqdm(tokenized_dataset):
            # Generate logits with the model.
            try:
                logits = model(
                    input_ids=torch.tensor(example['input_ids']).unsqueeze(0).to(DEVICE),
                    attention_mask=torch.tensor(example['attention_mask']).unsqueeze(0).to(DEVICE),
                    subject_token_ids=torch.tensor(example['subject_token_ids']).unsqueeze(0).to(DEVICE),
                    object_token_ids=torch.tensor(example['object_token_ids']).unsqueeze(0).to(DEVICE),
                    subject_token_ids_end=torch.tensor(example['subject_token_ids_end']).unsqueeze(0).to(DEVICE),
                    object_token_ids_end=torch.tensor(example['object_token_ids_end']).unsqueeze(0).to(DEVICE),
                    relation_token_ids=torch.tensor(example['relation_token_ids']).unsqueeze(0).to(DEVICE),
                )
            except:
                print(example, flush=True)

            # Convert the logits to binary predictions.
            predictions = torch.sigmoid(logits)
            predictions = (predictions >= .5).squeeze().detach().cpu().numpy().astype(int)

            # Get the id of the data example.
            id_ = example['id']

            # Initialize the relations list of the example, if not already done.
            if id_ not in outputs_by_id:
                outputs_by_id[id_] = []

            # Add relations when they are predicted.
            for k in range(len(predictions)):
                if predictions[k] == 1 and k != 0:
                    outputs_by_id[id_].append([int(example['subject_id']), ID2CLASS[k], int(example['object_id'])])

    # Complete with examples that contain no correct candidate entity pair.
    for id_ in initial_dataset_ids:
        if id_ not in outputs_by_id:
            outputs_by_id[id_] = []
    
    # Sort dictionary to restore correct order.
    outputs_by_id = {key: outputs_by_id[key] for key in initial_dataset_ids}

    # Create submission dataframe.
    data_dict = {'id' : list(outputs_by_id.keys()), 'relations' : list(outputs_by_id.values())}
    df_submission = pd.DataFrame(data_dict)

    # Convert relations attribute to have the strange string format.
    df_submission.relations = df_submission.relations.apply(str)

    # Save submission.
    if ONE_CLASS:
        df_submission.to_csv(f'_results/submission_{SUBMISSION_TYPE}_{ONE_CLASS_NAME}.csv', index=False)
    else:
        df_submission.to_csv(f'_results/submission_{SUBMISSION_TYPE}.csv', index=False)

if __name__ == '__main__':
    # Argument parsing.
    print("Argument parsing...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, required=True, help='Base path to the trained model.')
    parser.add_argument('--class_names', nargs='+', help='Names of the one-class classes.', default="")
    args = parser.parse_args()

    # Run script.
    for one_class_name in args.class_names:
        main(args.base_model_path, one_class_name=one_class_name)