import ast
import pandas as pd

SUBMISSION_PATH_1 = 'submission1.csv'
SUBMISSION_PATH_2 = 'submission2.csv'
OUTPUT_SUBMISSION_PATH = 'submission_merged.csv'

df_sub1 = pd.read_csv(SUBMISSION_PATH_1)
df_sub2 = pd.read_csv(SUBMISSION_PATH_2)

df_sub1['relations'] = df_sub1['relations'].apply(ast.literal_eval)
df_sub2['relations'] = df_sub2['relations'].apply(ast.literal_eval)

ids1 = df_sub1['id'].to_numpy()
ids2 = df_sub2['id'].to_numpy()
relations1 = df_sub1['relations'].to_numpy()
relations2 = df_sub2['relations'].to_numpy()

ids = []
relations = []

for i in range(len(df_sub1)):
    assert ids1[i] == ids2[i]
    ids.append(ids1[i])
    Ri1 = relations1[i]
    Ri2 = relations2[i]
    Ri = Ri1 + Ri2
    relations.append(Ri)

data_merged = {'id': ids, 'relations': relations}
df_sub_merged = pd.DataFrame(data=data_merged)
df_sub_merged.to_csv(OUTPUT_SUBMISSION_PATH, index=False)