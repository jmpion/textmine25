from openai import OpenAI
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers.data import DATA_PATH, load_test_data, load_train_data
from helpers.constants import CONFIG
from helpers.filter import validate_relation, ACTOR_ET, EVENT_ET, MATERIAL_ET, ORGANISATION_ET, PLACE_ET, QUANTITY_ET, TIME_ET, WEIGHT_ET, MATERIAL_REFERENCE_ET, COLOR_ET
from helpers.no_learning_utils import validate_relation_llm_yn

# LLM mode.
MODE = CONFIG['llm_yn']['mode'] #'openai' for OpenAI. 'huggingface' for HuggingFace models.
print(f"Selected LLM mode: {MODE}")
if MODE=='huggingface':
    # Load LLM for "Oui/Non" scheme.
    assert torch.cuda.is_available()
    MODEL_NAME = CONFIG['llm_yn']['llm_name']
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
elif MODE=='openai':
    client = OpenAI()
    tokenizer = None
    model = None

# Load the data.
TRAIN_OR_TEST = CONFIG['llm_yn']['train_or_test']
print(f"Train or test? --> {TRAIN_OR_TEST}")
if TRAIN_OR_TEST=='Train':
    df = load_train_data(DATA_PATH)
elif TRAIN_OR_TEST=='Test':
    df = load_test_data(DATA_PATH)

entities = df['entities'].to_numpy()
texts = df['text'].to_numpy()

# Both two variables below need be changed according to the targeted relation type.
target_eo_type = TIME_ET
target_r_type = 'WAS_CREATED_IN'
print(f"Targeted relation type: {target_r_type}")
eo_type_is_list = isinstance(target_eo_type, list)
relations = []
for i in tqdm(range(len(df))):
    Ei = entities[i]
    Ti = texts[i]
    predicted_relations = []

    for eo in Ei: # note here we start by iterating on the object entities.
        if eo_type_is_list:
            match_eo_type = (eo['type'] in target_eo_type)
        else:
            match_eo_type = (eo['type'] == target_eo_type)
        if match_eo_type:
            for es in Ei:
                if validate_relation(es['type'], target_r_type, eo['type']):
                    if validate_relation_llm_yn(Ti, es, eo, target_r_type, tokenizer=tokenizer, model=model, mode=MODE, client=client):
                        predicted_relations.append([es['id'], target_r_type, eo['id']])
    relations.append(predicted_relations)

data_dict = {"id" : df.index.to_numpy(), "relations" : relations}
new_df_sub = pd.DataFrame(data_dict)
new_df_sub.to_csv(f'results_llm_yn-{target_r_type}-{TRAIN_OR_TEST}.csv', index=False)