from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import ast
from helpers.data import convert_string_to_dict_entities, DATA_PATH, load_test_data
from helpers.filter import validate_relation
from helpers.constants import CONFIG

SUBMISSION_TYPE = CONFIG['submission_type']

NAME = '_results/submission_VALIDATION' # CONFIG['expe_name']
df_sub = pd.read_csv(f"{NAME}.csv")
df_sub['relations'] = df_sub['relations'].apply(ast.literal_eval)

relations = df_sub['relations'].to_numpy()
if SUBMISSION_TYPE=='VALIDATION':
    dataset = load_dataset('csv', data_files='data/train.csv')
    dataset = dataset.map(convert_string_to_dict_entities)
    dataset = dataset['train']
    train_eval_dataset = dataset.train_test_split(test_size=0.5, seed=0)
    dataset = train_eval_dataset['test']
    entities = dataset['entities']
elif SUBMISSION_TYPE=='TEST':
    df_entities = load_test_data(DATA_PATH)
    entities = df_entities['entities'].to_numpy()

valid_relations = []
for i in tqdm(range(len(df_sub))):
    entities_tmp = entities[i]
    relations_tmp = relations[i]
    valid_relations_tmp = []
    for relation in relations_tmp:
        es = relation[0]
        r = relation[1]
        eo = relation[2]

        es_type = entities_tmp[es]['type']
        eo_type = entities_tmp[eo]['type']

        if validate_relation(es_type, r, eo_type):
            valid_relations_tmp.append(relation)
    valid_relations.append(valid_relations_tmp)

data_dict = {"id" : df_sub['id'].to_numpy(), "relations" : valid_relations}
new_df_sub = pd.DataFrame(data_dict)
new_df_sub.to_csv(f"{NAME}_filtered.csv", index=False)
