import json
import warnings
import numpy as np
import cufflinks as cf

import os

import pandas as pd
from tqdm import tqdm
import openai
from langchain.llms import AzureOpenAI
import time
import re
from utils import clean,process,text_clean,count_word_num,truncate_text



"""
This script enhances the clarity of the original data and summarizes the AdviceDetails of all incidents using GPT4.
The final generated JSON file is stored in the 'Data_Under_Models' directory.

Before running this script, make sure to fill in the API key or set the API key as an environment variable.
Users can also customize the large language model according to their requirements.
"""


url = 'https://cloudgpt-dev.azurewebsites.net/api/cloud-gpt/scenario/raw-endpoint'

os.environ['CHATGPT_API_KEY'] = ''
oai_key = os.environ.get("CHATGPT_API_KEY")
openai.api_type = "azure"
openai.api_base = "https://cloudgpt.openai.azure.com/"
openai.api_key = oai_key
openai.api_version = "2023-03-15-preview"
chatgpt_model_name= "gpt-4-20230321"

cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


path = 'Data_Models/utf8allcharnobom.jsonl'
with open(path) as f:
    data = json.loads(str(f.read()).strip("'<>() "))  # .replace('\'', '\"')
df = pd.DataFrame(data)

df['AlertType'] = df['AlertType'].apply(lambda x: x.lower())
df.dropna(subset=['AdviceDetail'],inplace=True)

failed_prompt = []
df = clean(df, 'AdviceDetail')
df['Advice_Detail_Response'] = ''

for i in tqdm(range(len(df))):
    prompt = "Input:\n" + df['AdviceDetail'][i] + """\nContext: Please summarize the above input.
     Please note that the above input is a log information. The summary results should be about
      120 words no more than 140 words and should cover important information of the log as much 
      as possible, just return the summary without any additional output"""
    retry_count = 0
    max_retries = 10
    flag = False
    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=chatgpt_model_name,
                messages=[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                    {"role": "user", "content": prompt}
                ]
            )
            break
        except Exception as e:
            retry_count += 1
            print("{}th retry".format(retry_count))
            time.sleep(1)
            if retry_count > max_retries:
                flag = True
                break
    if flag:
        df['Advice_Detail_Response'][i] = df['AdviceDetail']
    else:
        df['Advice_Detail_Response'][i] = response['choices'][0]['message']['content']

df['Advice_Detail_Response'] = df.apply(lambda row: row['AdviceDetail'] if row['Advice_Detail_Response'] == '' else row['Advice_Detail_Response'], axis=1)
df['Advice_Detail_Response'] = df['Advice_Detail_Response'].apply(lambda x: truncate_text(str(x), 180) if count_word_num(str(x)) > 180 else x)
df['Advice_Detail_Response'] =df['Advice_Detail_Response'].replace('\n', '', regex=True)
df.to_json('Data_Models/all_info_gpt4_sum.jsonl',orient='records', lines=True)
