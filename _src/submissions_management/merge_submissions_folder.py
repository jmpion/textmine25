import ast
import os
import pandas as pd

SUBMISSIONS_DIR = 'submission_directory'
SUBMISSION_PATH = 'submission_after_merge.csv'

submissions_path = os.listdir(SUBMISSIONS_DIR)

for idx_path, submission_path in enumerate(submissions_path):
    if idx_path==0:
        submission_path_ref = os.path.join(SUBMISSIONS_DIR, submissions_path[0])
    else:
        submission_path_ref = SUBMISSION_PATH

    submission_path_add = os.path.join(SUBMISSIONS_DIR, submission_path)

    df_sub1 = pd.read_csv(submission_path_ref)
    df_sub2 = pd.read_csv(submission_path_add)

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
    df_sub_merged.to_csv(SUBMISSION_PATH, index=False)