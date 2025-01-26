import argparse
import logging
import numpy as np
from openai import OpenAI
from sklearn.model_selection import KFold
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers.data import load_train_data
from helpers.constants import CONFIG, OBJECT_TYPES
from helpers.filter import validate_relation
from helpers.no_learning_utils import validate_relation_llm_yn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
# Initialize the model based on mode
def initialize_model(mode):
    """Initialize the model and tokenizer based on the selected mode."""
    if mode == 'huggingface':
        model_name = CONFIG['llm_yn']['llm_name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        return tokenizer, model, None
    elif mode == 'openai':
        client = OpenAI()
        return None, None, client
    else:
        raise ValueError("Invalid mode. Must be 'huggingface' or 'openai'.")

# Define the cross-validation setup
def cross_validate(relation_type, data, tokenizer, model, client, mode, num_folds=5):
    """Perform cross-validation with relation validation."""
    target_r_type = relation_type # e.g., 'HAS_FOR_HEIGHT'
    target_eo_type = OBJECT_TYPES[relation_type]
    logger.info(f"Targeted relation type: {target_r_type}")

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    fold_metrics = []

    for fold, (_, val_idx) in enumerate(kf.split(data)):
        logger.info(f"Processing fold {fold + 1}/{num_folds}.")

        # Only evaluate on validation data.
        val_data = [data.iloc[i] for i in val_idx]

        # Forward pass with relation validation
        logger.info("Starting forward pass and relation validation...")
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for i, val_item in enumerate(val_data):
                entities = val_item['entities']
                text = val_item['text']
                relations = val_item.get('relations', [])

                predicted_relations = []
                for eo in entities:
                    if eo['type'] in target_eo_type:
                        for es in entities:
                            if validate_relation(es['type'], target_r_type, eo['type']):
                                is_valid = validate_relation_llm_yn(
                                    text, es, eo, target_r_type, tokenizer=tokenizer, model=model, mode=mode, client=client
                                )
                                if is_valid:
                                    predicted_relations.append([es['id'], target_r_type, eo['id']])

                true_relations = [r for r in relations if r[1]==target_r_type]
                tp += sum(1 for r in predicted_relations if r in true_relations)
                fp += sum(1 for r in predicted_relations if r not in true_relations)
                fn += sum(1 for r in true_relations if r not in predicted_relations)

        # Compute fold metrics
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        fold_metrics.append({'precision': precision, 'recall': recall, 'f1': f1})
        logger.info(f"Fold {fold + 1} Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Aggregate metrics across all folds
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])

    logger.info(f"Aggregate F1 Metrics: Average F1={avg_f1:.4f}, Std F1={std_f1:.4f}")

# Main execution
def main():
    parser = argparse.ArgumentParser(description="Cross-validation script with relation validation.")
    parser.add_argument('--relation_type', type=str, required=True, help="Targeted relation type.")
    parser.add_argument('--mode', type=str, choices=['huggingface', 'openai'], required=True, help="Specify the model mode.")
    parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for cross-validation.")

    args = parser.parse_args()

    data = load_train_data('data')
    tokenizer, model, client = initialize_model(args.mode)

    cross_validate(args.relation_type, data, tokenizer, model, client, args.mode, args.num_folds)

if __name__ == '__main__':
    main()
