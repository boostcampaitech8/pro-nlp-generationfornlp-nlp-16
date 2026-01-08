import pandas as pd
import collections

files = [
    'model_0.7590.csv',
    'model_0.7779.csv', 
    'model_0.7791.csv', 
    'model_0.7793.csv', 
    'model_0.7810.csv', 
    'model_0.7828.csv'
]

strongest_model = 'model_0.7828.csv'
expert_model = 'model_0.7791.csv'

dfs = [pd.read_csv(f) for f in files]
all_preds = pd.DataFrame()
all_preds['id'] = dfs[0]['id']

for i, f in enumerate(files):
    df_temp = dfs[i].set_index('id')
    all_preds[f] = all_preds['id'].map(df_temp['answer'])

def get_power_vote(row):
    votes = [row[f] for f in files]
    counts = collections.Counter(votes)
    most_common = counts.most_common()
    
    top_vote, top_count = most_common[0]
    
    if top_count >= 4:
        return top_vote
    
    val_strongest = row[strongest_model]
    val_expert = row[expert_model]
    
    if val_strongest == val_expert:
        return val_strongest
    else:
        return val_strongest

all_preds['final_answer'] = all_preds.apply(get_power_vote, axis=1)

submission = all_preds[['id', 'final_answer']].rename(columns={'final_answer': 'answer'})
submission.to_csv('power_ensemble_output.csv', index=False)

print(f"Power Ensemble Completed.")
print(f"Strategy: Majority Vote (>=4) -> ({strongest_model} == {expert_model}) -> {strongest_model}")
print(submission.head())
