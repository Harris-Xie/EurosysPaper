from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import json
import pandas as pd

from sklearn.model_selection import ParameterGrid
params = {'num': [3,5,9,12,15], 'alpha': [0.001,0.01,0.1,0.3,0.7,0.9]}
param_grid=ParameterGrid(params)
res=[]
for param in param_grid:
    with open('Data_Models/new_result/output_{}_{}.json.'.format(param['alpha'],param['num']), "r") as f:
        y_pred = f.read()
    y_pred=json.loads(y_pred)

    with open('Data_Models/output_true.json', "r") as f:
        y_true = f.read()
    y_true=json.loads(y_true)


    length = len(y_true)
    train = pd.read_json('Data_Models/train.jsonl', lines=True)
    train_set = set(train['Keyword'])
    for i in range(length):
        if y_true[i] not in train_set:
            train_set.add(y_true[i])
            y_true[i] = -1
            y_pred[i] = -1
    y_pred = [x for x in y_pred if x != -1]
    y_true = [x for x in y_true if x != -1]

    acc=accuracy_score(y_true, y_pred)
    recall=recall_score(y_true, y_pred, average='weighted')
    precision=precision_score(y_true, y_pred, average='weighted')
    f1_micro = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print({'alpha':param['alpha'],'k':param['num'],'acc':acc,'Weighted_F1':f1_micro,'F1_macro':f1_macro, 'Weighted_Recall':recall,'Weighted_Precision':precision})
    res.append({'alpha':param['alpha'],'k':param['num'],'F1_micro':acc,'Weighted_F1':f1_micro,'F1_macro':f1_macro,'Weighted_Recall':recall,'Weighted_Precision':precision})
results=json.dumps(res)
with open('Data_Models/new_result/output_grid_search.json', 'w') as f:
    f.write(results)

# with open('Data_Models/new_result/output_{}_{}.json.'.format(0.3,5), "r") as f:
#     y_pred = f.read()
# y_pred=json.loads(y_pred)
#
# with open('Data_Models/output_true.json', "r") as f:
#     y_true = f.read()
# y_true=json.loads(y_true)
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
#
# acc=accuracy_score(y_true, y_pred)
# recall=recall_score(y_true, y_pred, average='weighted')
# precision=precision_score(y_true, y_pred, average='weighted')
# f1_micro = f1_score(y_true, y_pred, average='weighted')
# f1_macro = f1_score(y_true, y_pred, average='macro')
#
# print({'F1_micro':acc,'Weighted_F1':f1_micro,'F1_macro':f1_macro, 'Weighted_Recall':recall,'Weighted_Precision':precision})
