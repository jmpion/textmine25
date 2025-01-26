from helpers.constants import CONFIG, ONE_CLASS, initialize_classes
from helpers.spert import SpERT
from transformers import AutoModelForSequenceClassification

SPERT = CONFIG['spert']

def load_model(num_classes: int, model_path: str, one_class_name: str = ""):
    _, CLASS2ID, ID2CLASS = initialize_classes(one_class_name)
    if SPERT == 'Yes':
        model = SpERT(
            bert_model_name=model_path,
            hidden_size=768,
            num_labels=2,
        )
    else:
        if not ONE_CLASS:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_classes,
                id2label=ID2CLASS,
                label2id=CLASS2ID,
                problem_type='multi_label_classification',
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_classes,
                id2label=ID2CLASS,
                label2id=CLASS2ID,
            )
    return model
