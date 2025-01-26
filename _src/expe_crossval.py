import argparse
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, set_seed, TrainingArguments
from sklearn.model_selection import KFold

# Importations personnelles.
from helpers.constants import CONFIG, initialize_classes
from helpers.customtrainer import CustomTrainer
from helpers.data import prepare_relation_datasets, load_preprocess_function, prepare_dataset
from helpers.entity_markers import additional_tokens_from_wrapper_mode
from helpers.evaluation import compute_metrics
from helpers.model import load_model

def main(output_dir, one_class_name=""):
    # Reproducibility.
    set_seed(0, deterministic=False)  # deterministic=True pour déterminisme complet, mais plus lent.

    # Obtention du label de la classe de relations.
    assert one_class_name != "", "Please provide the name of the one-class class."
    print(f"\nOne class: {one_class_name}\n", flush=True)

    # Initialisation des classes.
    CLASSES, CLASS2ID, ID2CLASS = initialize_classes(one_class_name)
    NUM_CLASSES = len(CLASSES)

    # Obtention du nom du modèle et du tokenizer.
    base_model_path = CONFIG['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Chargement et préparation du jeu de données.
    dataset = prepare_dataset('data/train.csv')
    dataset = dataset['train']
    
    # Préparation de la validation croisée.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_metrics = []

    # Itération sur chaque pli.
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nStarting Fold {fold + 1}...")

        # On charge le modèle une fois par pli.
        model = load_model(NUM_CLASSES, base_model_path, one_class_name=one_class_name)

        # Séparation en entraînement et validation.
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        # Transformation des jeux de données, spécifiquement pour la tâche de classification binaire.
        train_dataset_new, val_dataset_new = prepare_relation_datasets(train_dataset, val_dataset, one_class_name)

        # Marquage d'entités.
        ENTITY_MARKERS_ON = CONFIG['entity_markers']['entity_markers_on']
        if ENTITY_MARKERS_ON=='Yes':
            # Add special tokens (entity markers) to the tokenizer.
            WRAPPER_MODE =  CONFIG['entity_markers']['wrapper_mode']
            added_special_tokens = additional_tokens_from_wrapper_mode(WRAPPER_MODE)
            special_tokens = {'additional_special_tokens': added_special_tokens}
            tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer) + 1)

        # Prétraitement du jeu de données.
        preprocess_function = load_preprocess_function(tokenizer, CLASSES, CLASS2ID, one_class_name)
        train_tokenized_dataset = train_dataset_new.map(preprocess_function)
        val_tokenized_dataset = val_dataset_new.map(preprocess_function)

        # Dans le cas d'une classe avec peu de données (comme HAS_LATITUDE et HAS_LONGITUDE)
        # il peut arriver que le jeu de données de validation soit vide.
        if len(val_tokenized_dataset) == 0:
            print(f"Fold {fold + 1} has empty validation set.")
            continue

        # Initialize training arguments
        training_args = TrainingArguments(
            eval_strategy='epoch',
            fp16=True,
            learning_rate=5e-5,
            load_best_model_at_end=True,
            logging_dir='./logs',
            lr_scheduler_type='cosine',
            metric_for_best_model='eval_f1_macro',
            num_train_epochs=20,
            output_dir=f"{output_dir}/{one_class_name}_fold_{fold + 1}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_strategy='epoch',
            save_total_limit=2,
            warmup_steps=0,
            weight_decay=0.01,
        )

        # Pour geler des couches du réseau.
        # Par défaut, après plusieurs tentatives, aucune couche n'est gelée.
        FREEZE = False
        if FREEZE:
            # Gel les n premières couches.
            n_layers_to_freeze = 11 # Défaut: 11.
            for name, param in model.named_parameters():
                if any(f"layer.{i}." in name for i in range(n_layers_to_freeze)):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # Module d'entraînement.
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_dataset,
            eval_dataset=val_tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=5),
            ]
        )

        # Entraînement et évaluation pour ce pli.
        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        print(f"Fold {fold + 1} metrics: {metrics}")

    # Calcul de la moyenne et de la déviation standard à travers les plis.
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
    std_metrics = {metric: np.std([fold[metric] for fold in fold_metrics], ddof=1) for metric in fold_metrics[0]}  # Using sample standard deviation

    print("\nAverage metrics across folds:")
    print(avg_metrics)

    print("\nStandard deviation of metrics across folds:")
    print(std_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Base path to the output directory for the trained model.')
    parser.add_argument('--class_names', nargs='+', help='Names of the one-class classes.', default="")
    args = parser.parse_args()

    # Exécuter le script, pour chaque classe spécifiée.
    for one_class_name in args.class_names:
        main(args.output_dir, one_class_name=one_class_name)
