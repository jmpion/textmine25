import ast
import pandas as pd
from tqdm import tqdm

SUBMISSION_PATH_1 = 'submission_to_inject.csv'
SUBMISSION_PATH_2 = 'submission_to_receive.csv'
OUTPUT_SUBMISSION_PATH = 'submission_substituted.csv'
CLASS_1_to_2 = 'HAS_FOR_LENGTH' # the class to substitute from 1, into 2.
# All relations from this relation type in submission 2 are removed.
# All relations from this relation type in submission 1 are substituted into submission 2, to create the new submission.

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

for i in tqdm(range(len(df_sub1))):
    assert ids1[i] == ids2[i]
    ids.append(ids1[i])
    Ri1 = relations1[i]
    Ri2 = relations2[i]
    Ri = []
    for r1 in Ri1:
        rt1 = r1[1]
        if rt1 == CLASS_1_to_2:
            Ri.append(r1)
        else:
            pass
    for r2 in Ri2:
        rt2 = r2[1]
        if rt2 == CLASS_1_to_2:
            pass
        else:
            Ri.append(r2)
    relations.append(Ri)

data_merged = {'id': ids, 'relations': relations}
df_sub_merged = pd.DataFrame(data=data_merged)
df_sub_merged.to_csv(OUTPUT_SUBMISSION_PATH, index=False)