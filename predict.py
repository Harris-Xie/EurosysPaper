import time
import json
import fasttext
import pandas as pd
import os
import faiss
import re
import numpy as np
from tqdm import tqdm
import openai
from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.model_selection import ParameterGrid
from utils import truncate_text,clean,count_word_num,process,text_clean
from fuzzywuzzy import fuzz
url = 'https://cloudgpt-dev.azurewebsites.net/api/cloud-gpt/scenario/raw-endpoint'

os.environ['CHATGPT_API_KEY'] = ''
oai_key = os.environ.get("CHATGPT_API_KEY")
openai.api_type = "azure"
openai.api_base = "https://cloudgpt.openai.azure.com/"
openai.api_key = oai_key
openai.api_version = "2023-03-15-preview"
chatgpt_model_name= "gpt-4-20230321"

"""
This file primarily processes web application requests and manages the flow of data processing.

The <OWL> method is our advanced approach.
The <GPT_predict> method generates output results using the GPT model.
The <onlyGPT> method is for ablation experiment, using only prompt to make prediction
The <onlyRetrieval> method is for ablation experiment, only return the most similar incident.

"""



def GPT_predict(alpha,num):

    column_list=['Advice_Detail_Response']

    train_data = pd.read_json('Data_Models/GPT4-sum/train.jsonl', lines=True)
    model = fasttext.load_model('Data_Models/models')
    test_data=pd.read_json('Data_Models/GPT4-sum/test.jsonl',lines=True)
    test_data['CreatedTime'] = pd.to_datetime(test_data['CreatedTime'])
    train_data['CreatedTime'] = pd.to_datetime(train_data['CreatedTime'])
    train_data = train_data.sort_values('CreatedTime', ascending=True).reset_index(drop=True)
    test_data=test_data.sort_values('CreatedTime', ascending=True).reset_index(drop=True)
    y_true=[]
    for i in range(len(test_data)):
        y_true.append(test_data['Keyword'][i])
    y_true_str = json.dumps(y_true)
    with open('Data_Models/GPT4-sum/output_true.json', 'w') as f:
        f.write(y_true_str)
    y_pred=[]
    y_generator={}
    #store the index
    filtered_data = train_data
    filtered_data["metadata"] = filtered_data.to_dict(orient="records")
    filtered_data_text = filtered_data['metadata'].apply(lambda x: text_clean(
        str(x['KeyValuePairs']) + ' ' + x['AlertType'] + ' ' + str(x['ScopeType']) + ' ' + x['AdviceDetail']))
    vectors = np.array([model.get_sentence_vector(text) for text in filtered_data_text])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    log_output=[]
    def decay_factor(alpha, time_diff):
        return np.exp(-alpha * time_diff)

    def similarity_with_time_decay(x, length, created_time,AlertType, alpha ,num):
        #distance
        distances, indices = index.search(x, k=length)
        distances=distances.flatten()
        sorted_indices = np.argsort(indices)
        sorted_distances = distances[sorted_indices]
        # time_diff
        time_diffs = (created_time - filtered_data['CreatedTime']).values / np.timedelta64(1, 'D')
        AlertType_scores = [fuzz.partial_ratio(AlertType, alert) for alert in filtered_data['AlertType']]
        # decay
        decay_factors = decay_factor(alpha, time_diffs)
        # similarity
        similarities = 1 / (1 + sorted_distances) * decay_factors* AlertType_scores
        # find k similar
        sorted_indices = np.argsort(similarities[0])[::-1]
        return filtered_data.iloc[sorted_indices[:num]]

    for i in tqdm(range(len(test_data))):
        if i != 31 and i!=44 and i!=59 and i!= 62 and i!=99 and i!=122 and i!=141:
            continue
        new_prompt ="Prompt: Incident Root Cause Identification\n"
        new_prompt += """Context:The following input displays error log information related to an incident. Your task is to select the incident information that is most likely to have the same root cause keyword. You will be provided with several options, including other similar log information and a keyword. If you cannot find a matching option, please return -1 and generate a new precise root cause keyword without using words that already exist in input. Provide your explanation along with the selected option in JSON format, for example: {"Option": "B", "Explanation": ...}.\n"""
        new_prompt +="""Task: Select the incident information that is most likely to have the same root cause keyword as the input and provide an explanation.\n"""
        new_prompt += "Input:\n"
        for item in column_list:
            new_prompt += item + ':' + str(test_data[item][i]) + ' '
        # new_prompt+=truncate_text(test_data['AdviceDetail'][i],180)
        new_prompt+='\n'
        created_time = test_data['CreatedTime'][i]
        AlertType=test_data['AlertType'][i]
        filtered_data = train_data
        query = text_clean(str(test_data['KeyValuePairs'][i]) + ' ' + test_data['AlertType'][i] + ' ' + str(test_data['ScopeType'][i]) + ' ' +test_data['AdviceDetail'][i])
        query_vector = np.array([model.get_sentence_vector(query)])
        train_samples=similarity_with_time_decay(query_vector,len(filtered_data),created_time,AlertType,alpha,num)
        # train_samples = train_samples.drop_duplicates('Keyword')
        order = ['A','B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q']
        map_dict = {}
        new_prompt += 'Options:\n'
        j=0
        similar_Id=[]
        for _, train_sample in train_samples.iterrows():
            similar_Id.append(train_sample['AlertId'])
            new_prompt+=order[j] + ':'+' '
            for item in column_list:
                new_prompt += item+':'+str(train_sample[item])+' '
            # new_prompt += truncate_text(test_data['AdviceDetail'][i], 180)
            new_prompt+='Keyword:'+str(train_sample['Keyword'])
            new_prompt+='\n'
            map_dict[order[j]] = {'AlertId':train_sample['AlertId'],'Keyword':train_sample['Keyword']}
            j += 1
        new_prompt+="""Remember to provide your answer and explanation in JSON format, for example: {"Option": "B", "Explanation": ...}.\n"""
        new_prompt += 'Answer:'
        retry_count = 0
        max_retries = 10

        flag=False
        while True:
            try:
                response=openai.ChatCompletion.create(
                    engine=chatgpt_model_name,
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": new_prompt}
                    ]
                )
                break
            except Exception as e:
                print(e)
                retry_count+=1
                print("{}th retry".format(retry_count))
                time.sleep(1)
                if retry_count>max_retries:
                    flag=True
                    break
        if flag:
            keyword = 'None'
            y_pred.append(keyword)
            row_to_add = test_data.iloc[i]
            train_data = pd.concat([train_data, row_to_add.to_frame().T], axis=0, ignore_index=True)
            index.add(query_vector)
            continue
        res=response['choices'][0]['message']['content']
        match = re.search(r'\{.*?\}', res, flags=re.DOTALL)
        if match:
            try:
                json_dict = json.loads(match.group())
                option = json_dict['Option']
                if option[0]=='-':
                    option=option[:2]
                else:
                    option = option[0]#to avoid muiltple answers


            except:
                match = re.search(r'"Option":\s*"(\w)\b', res)
                if match:
                    option = match.group(1)
        else:
            words=res.split()
            option=words[0][0]
        if option=='-1' or option=='None':
            keyword='Others'
            y_generator[i] = res
        else:
            try:
                keyword = map_dict[option]['Keyword']
                AlertId=map_dict[option]['AlertId']
            except:
                keyword='Others'
                AlertId=-1
        log_output.append({'AlertId':str(test_data['AlertId'][i]),'Info':{'Prompt':new_prompt,'similarId':str(similar_Id),'ChosenAlertId':str(AlertId),'raw_res':res}})
        y_pred.append(keyword)
        row_to_add = test_data.iloc[i]
        train_data = pd.concat([train_data, row_to_add.to_frame().T], axis=0, ignore_index=True)
        index.add(query_vector)


    json_gen = json.dumps(y_generator)
    with open('Data_Models/GPT4-sum/output_generator_{}_{}_fixed.json'.format(alpha, num), 'w') as f:
        f.write(json_gen)
    return 0


def onlyGPT():
    train_data = pd.read_json('Data_Models/train.jsonl', lines=True)
    keyword_list=list(train_data['Keyword'].unique())
    test_data = pd.read_json('Data_Models/test_summarized.jsonl', lines=True)
    y_true = []
    y_pred=[]

    for i in tqdm(range(len(test_data))):
        y_true.append(test_data['Keyword'][i])
        new_prompt="Input:"+test_data['AdviceDetail'][i]+'\n'
        new_prompt+="""This is an error log message. Please help me analyze and select one and only the most likely keyword, The list of keywords is as follows:\n"""
        new_prompt+='['
        for word in keyword_list:
            new_prompt+=word+' '
        new_prompt+=']'+'\n'+'Answer:'+'\n'
        retry_count = 0
        max_retries = 50
        while True:
            try:
                response=openai.ChatCompletion.create(
                    engine=chatgpt_model_name,
                    messages=[
                        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                        {"role": "user", "content": new_prompt}
                    ]
                )
                break
            except Exception as e:
                retry_count+=1
                print("{}th retry".format(retry_count))
                time.sleep(1)
                if retry_count>max_retries:
                    raise e

        res=response['choices'][0]['message']['content']
        y_pred.append(res.split()[0])
        if test_data['Keyword'][i] not in keyword_list:
            keyword_list.append(test_data['Keyword'][i])
    json_str = json.dumps(y_pred)
    with open('Data_Models/new_result/output_OnlyGPT.json', 'w') as f:
        f.write(json_str)

def OnlyRetrieval(alpha,num):
    column_list = ['Advice_Detail_Response']
    train_data = pd.read_json('Data_Models/train.jsonl', lines=True)
    model = fasttext.load_model('Data_Models/train_models')
    test_data = pd.read_json('Data_Models/test_summarized.jsonl', lines=True)
    test_data['CreatedTime'] = pd.to_datetime(test_data['CreatedTime'])
    train_data['CreatedTime'] = pd.to_datetime(train_data['CreatedTime'])
    train_data = train_data.sort_values('CreatedTime', ascending=True).reset_index(drop=True)
    test_data = test_data.sort_values('CreatedTime', ascending=True).reset_index(drop=True)
    y_true = []
    for i in range(len(test_data)):
        y_true.append(test_data['Keyword'][i])
    y_true_str = json.dumps(y_true)
    with open('Data_Models/output_true.json', 'w') as f:
        f.write(y_true_str)
    y_pred = []
    # store the index
    filtered_data = train_data
    filtered_data["metadata"] = filtered_data.to_dict(orient="records")
    filtered_data_text = filtered_data['metadata'].apply(lambda x: text_clean(
        str(x['KeyValuePairs']) + ' ' + x['AlertType'] + ' ' + str(x['ScopeType']) + ' ' + x['AdviceDetail']))
    vectors = np.array([model.get_sentence_vector(text) for text in filtered_data_text])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    def decay_factor(alpha, time_diff):
        return np.exp(-alpha * time_diff)

    def similarity_with_time_decay(x, length, created_time, alpha, num):
        # distance
        distances, indices = index.search(x, k=length)
        distances = distances.flatten()
        sorted_indices = np.argsort(indices)
        sorted_distances = distances[sorted_indices]
        # time_diff
        time_diffs = (created_time - filtered_data['CreatedTime']).values / np.timedelta64(1, 'D')
        # decay
        decay_factors = decay_factor(alpha, time_diffs)
        # similarity
        similarities = 1 / (1 + sorted_distances) * decay_factors
        # find k similar
        sorted_indices = np.argsort(similarities[0])[::-1]
        return filtered_data.iloc[sorted_indices[:num]]

    for i in tqdm(range(len(test_data))):
        created_time = test_data['CreatedTime'][i]
        filtered_data = train_data
        query = text_clean(str(test_data['KeyValuePairs'][i]) + ' ' + test_data['AlertType'][i] + ' ' + str(
            test_data['ScopeType'][i]) + ' ' + test_data['AdviceDetail'][i])
        query_vector = np.array([model.get_sentence_vector(query)])
        train_samples = similarity_with_time_decay(query_vector, len(filtered_data), created_time, alpha, num)
        train_samples = train_samples.drop_duplicates('Keyword')
        train_samples=train_samples.reset_index(drop=True)
        Keyword=train_samples['Keyword'][0]
        y_pred.append(Keyword)
        row_to_add = test_data.iloc[i]
        train_data = pd.concat([train_data, row_to_add.to_frame().T], axis=0, ignore_index=True)
        index.add(query_vector)

    json_str = json.dumps(y_pred)
    with open('Data_Models/new_result/output_OnlyRetrieval.json'.format(alpha, num), 'w') as f:
        f.write(json_str)

def OWL():
    params = {'num': [5], 'alpha': [0.3]}
    param_grid=ParameterGrid(params)
    results=[]
    for param in param_grid:
        result=GPT_predict(param['alpha'],param['num'])
        print(result)
        results.append(result)
    print(results)
    results=json.dumps(results)
    with open('Data_Models/GPT4-sum/output_grid_search.json', 'w') as f:
        f.write(results)


if __name__=="__main__":
    OWL()
    onlyGPT()
    OnlyRetrieval(0.3,1)
