import argparse
import ast
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from transformers import set_seed

from helpers.constants import CONFIG, RELATIONS_IDS
from helpers.data import convert_string_to_dict_entities, convert_string_to_dict_relations
from helpers.evaluation import compute_metrics_from_scores

# Reproducibility.
set_seed(0)

# Add arguments.
parser = argparse.ArgumentParser(description="Parser for validation.py")
parser.add_argument('--submission_file', '-s', type=str, help="Path to the submission file.")

# Parse the arguments.
args = parser.parse_args()

# Data handler.
NAME = args.submission_file # E.g., '_results/submission_VALIDATION.csv'
NUM_CLASSES = len(RELATIONS_IDS) + 1

# Submission data.
df_submission = pd.read_csv(NAME)
df_submission['relations'] = df_submission['relations'].apply(ast.literal_eval)

# Reference data.
dataset = load_dataset('csv', data_files='data/train.csv')

# Convert dataset string columns to Python objects.
dataset = dataset.map(convert_string_to_dict_entities)
dataset = dataset.map(convert_string_to_dict_relations)

# Access subdataset.
dataset = dataset['train']

# TODO: check what the difference is between the line below and the current code above.
# dataset = load_train_data(DATA_PATH)

relation_to_scores = {relation_type: {'TP': 0, 'FP': 0, 'FN': 0} for relation_type in RELATIONS_IDS}

predictions = df_submission['relations'].to_numpy()
reference = dataset['relations']
reference = [[[int(r[0]), r[1], int(r[2])] for r in Ri] for Ri in reference]

assert isinstance(reference, list)
assert set(dataset['id']) == set(df_submission['id'].tolist()), print("Dataset[id]:", dataset['id'], "\nSubmission[id]:", df_submission['id'].tolist())

predictions = predictions[:min(len(predictions), len(reference))]
reference = reference[:min(len(predictions), len(reference))]
for i in tqdm(range(len(predictions))):
    predicted_relations = predictions[i]
    reference_relations = reference[i]

    try:
        for relation in predicted_relations:
            type_ = relation[1]
            if relation in reference_relations:
                relation_to_scores[type_]['TP'] += 1
            else:
                relation_to_scores[type_]['FP'] += 1
        
        for relation in reference_relations:
            type_ = relation[1]
            if relation not in predicted_relations:
                relation_to_scores[type_]['FN'] += 1
    except Exception as e:
        print(e)

relation_to_metrics = compute_metrics_from_scores(relation_to_scores)
print(relation_to_metrics)

# Function to round values in the dictionary
rounded_data = {key: {sub_key: round(value, 3) for sub_key, value in sub_dict.items()} for key, sub_dict in relation_to_metrics.items()}

# Print the rounded dictionary
print(rounded_data)

total_f1 = 0
for rt in relation_to_metrics:
    total_f1 += relation_to_metrics[rt]['F1']
avg_f1 = total_f1 / len(relation_to_metrics)
print(f"{avg_f1:.2%}")

df_results = pd.DataFrame(data=rounded_data)
df_results.to_excel(f'{NAME[:-4]}_metrics.xlsx')