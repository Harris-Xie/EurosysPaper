import pandas as pd
import json
df=pd.read_json('Data_Models/GPT4-sum/all_info_gpt4_sum.jsonl',lines=True)
df_sorted=df.sort_values('CreatedTime', ascending=False)
n = len(df_sorted)
n_1 = int(n * 0.25)
n_2 = n - n_1
df_test = df_sorted[:n_1]
df_train = df_sorted[n_1:]
df_train = df_train.sort_values('CreatedTime', ascending=True).reset_index(drop=True)
df_test=df_test.sort_values('CreatedTime', ascending=True).reset_index(drop=True)
df_train.to_json('Data_Models/GPT4-sum/train.jsonl', orient='records', lines=True)
dict=df_test.iloc[1].to_dict()
json_data = json.dumps(dict)
with open("sample_data.json", "w") as file:
    file.write(json_data)
df_test.to_json('Data_Models/GPT4-sum/test.jsonl', orient='records', lines=True)
