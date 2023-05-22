import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
from utils import text_clean
import fasttext

# Read the dataset
df = pd.read_json('Data_Models/train.jsonl', lines=True)
test = pd.read_json('Data_Models/test.jsonl', lines=True)
y_true = test['Keyword'].tolist()

# Convert text data to numerical feature vectors
ebd = fasttext.load_model('Data_Models/train_models')
df['metadata'] = df.to_dict(orient="records")
document_list = df['metadata']
train_texts = document_list.apply(lambda x: text_clean(str(x['KeyValuePairs']) + ' ' + x['AlertType'] + ' ' + str(x['ScopeType']) + ' ' + x['AdviceDetail']))
X = np.array([ebd.get_sentence_vector(text) for text in train_texts])
y = df['Keyword'].tolist()

# Convert labels to numerical encoding
label_encoder = LabelEncoder()
y_u = set(y)
label_dict = {}
for item in y_u:
    # If the element has not appeared in the dictionary yet
    if item not in label_dict:
        # Add it to the dictionary and assign a new numerical label to it
        label_dict[item] = len(label_dict)

y = [label_dict[item] for item in y]

xgb_model = xgb.XGBClassifier(objective='multi:softmax', max_depth=7, learning_rate=0.1, n_estimators=500)
xgb_model.fit(X, y)

# Use the trained model to predict the test set and calculate the prediction accuracy
test = pd.read_json('Data_Models/test.jsonl', lines=True)

test['metadata'] = test.to_dict(orient="records")
y_true_id = []
for item in y_true:
    if item in y_u:
        y_true_id.append(label_dict[item])
    else:
        y_true_id.append(-1)
document_list = test['metadata']
test_texts = document_list.apply(lambda x: text_clean(str(x['KeyValuePairs']) + ' ' + x['AlertType'] + ' ' + str(x['ScopeType']) + ' ' + x['AdviceDetail']))  # str(x['KeyValuePairs'])+' '+
text = np.array([ebd.get_sentence_vector(text) for text in test_texts])
y_pred = xgb_model.predict(text)
f1_micro = f1_score(y_pred, y_true_id, average='micro')
f1_macro = f1_score(y_pred, y_true_id, average='macro')
print('f1_micro', f1_micro)
print('f1_macro', f1_macro)
