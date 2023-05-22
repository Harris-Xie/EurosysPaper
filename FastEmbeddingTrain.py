import fasttext
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import cufflinks as cf
from transformers import AutoTokenizer
import warnings
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd
import numpy as np
cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
from utils import text_clean,trainingFile_preprocess

"""
Train a Fasttext Model
"""

df = pd.read_json('Data_Models/GPT4-sum/train.jsonl', lines=True)
df['metadata']=df.to_dict(orient="records")
train = df
train = train.sort_values(by=['AlertId'])
# use ICL from train to construct the test dataset by recalling samples in the training dataset


file=trainingFile_preprocess(train)
np.savetxt('Data_Models/GPT4-sum/train_tem.txt',file,fmt='%s')
model = fasttext.train_supervised(input='Data_Models/GPT4-sum/train_tem.txt',epoch=200,lr=0.9,thread=7,wordNgrams=2,seed=41)
# test=pd.read_json('Data_Models/test.jsonl', lines=True)
# test['metadata']=test.to_dict(orient="records")
# y_true = test['Keyword'].tolist()
# document_list = test['metadata']
# test_texts = document_list.apply(lambda x: text_clean(str(x['KeyValuePairs']) + ' ' + x['AlertType'] + ' ' + str(x['ScopeType'])  + ' ' + x['AdviceDetail']))  # str(x['KeyValuePairs'])+' '+
# y_pred=[]
# for text in test_texts:
#     pred_label = model.predict(text)[0][0]
#     y_pred.append(pred_label[len('__label__'):])
# categs=list(set(y_pred))
# # draw_confusion_matrix(y_true, y_pred, categs, figsize=(10, 10))
#
#
# length = len(y_true)
# train = pd.read_json('Data_Models/train.jsonl', lines=True)
# train_set = set(train['Keyword'])
# for i in range(length):
#     if y_true[i] not in train_set:
#         train_set.add(y_true[i])
#         y_true[i] = -1
#         y_pred[i] = -1
# y_pred = [x for x in y_pred if x != -1]
# y_true = [x for x in y_true if x != -1]
#
# acc = accuracy_score(y_true, y_pred)
# f1_micro = f1_score(y_true, y_pred, average='micro')
# acc=accuracy_score(y_true, y_pred)
# f1_macro = f1_score(y_true, y_pred, average='macro')
# print('acc:',acc)
# print('f1_micro:',f1_micro)
# print('f1_macro:',f1_macro)

model.save_model('Data_Models/train_models')

