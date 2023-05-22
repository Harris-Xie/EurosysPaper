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
from tqdm import tqdm
import numpy as np
import json
cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
from utils import text_clean,trainingFile_preprocess

df = pd.read_json('Data_Models/train.jsonl', lines=True)
all_keywords = df['Keyword'].unique().tolist()
df['metadata']=df.to_dict(orient="records")
train = df
train = train.sort_values(by=['AlertId'])

test=pd.read_json('Data_Models/test.jsonl', lines=True)
test['metadata']=test.to_dict(orient="records")
test = test.sort_values(by=['AlertId'])
y_true = test['Keyword'].tolist()
y_pred=[]
for i in tqdm(range(len(test))):
    train = train.sort_values(by=['AlertId'])
    file = trainingFile_preprocess(train)
    np.savetxt('Data_Models/train_tem.txt', file, fmt='%s')
    model = fasttext.train_supervised(input='Data_Models/train_tem.txt', epoch=200, lr=0.9, thread=7, wordNgrams=2,seed=41)
    text = text_clean(str(test['KeyValuePairs'][i]) + ' ' + test['AlertType'][i] + ' ' + str(test['ScopeType'][i]) + ' ' +test['AdviceDetail'][i])
    pred_label = model.predict(text)[0][0]
    y_pred.append(pred_label[len('__label__'):])
    row_to_add = test.iloc[i]
    train = pd.concat([train, row_to_add.to_frame().T], axis=0, ignore_index=True)


json_str=json.dumps(y_pred)
with open('Data_Models/FastPred.json', 'w') as f:
    f.write(json_str)

y_pred=['TenantIsRelocated', 'SelfRecover', 'LDAP-related-issue', 'TenantIsRelocated', 'jpnprd01_TYC_DC', 'BadADServerObjects', 'FHSSpareServerFail', 'Unhandled_DCOverloadedException', 'Unhandled_DCOverloadedException', 'Auth_EmptyEPKClaim', 'XSOLogging_Regression', 'HubPortExhaustion', 'XSOLogging_Regression', 'ItemAssistants', 'HubPortExhaustion', 'NetworkIssue', 'NetworkIssue', 'HubPortExhaustion', 'HubPortExhaustion', 'HubPortExhaustion', 'SubmissionQueueStuck_SingleServer', 'SelfRecover', 'FastTrain', 'AddressBookPolicy_corrupt', 'SelfRecover', 'SubmissionQueueStuck_SingleServer', 'LDAP-related-issue', 'FastTrain', 'SelfRecover', 'jpnprd01_TYC_DC', 'CATExpander_RunningSlow_SingleServer', 'CATExpander_RunningSlow_SingleServer', 'TenantCustomizationEnabled', 'CAT_Processing', 'CATExpander_RunningSlow_SingleServer', 'CATExpander_RunningSlow_SingleServer', 'CATExpander_RunningSlow_SingleServer', 'CATExpander_RunningSlow_SingleServer', 'MdbReplication', 'CATExpander_RunningSlow_SingleServer', 'Setting-changes', 'CATExpander_RunningSlow_SingleServer', 'FalseAlarm', 'GetBricksV2ServiceEndpointHang', 'MDBReplication', 'MissingAuthoritativeDomain', 'HubPortExhaustion', 'HubPortExhaustion', 'ItemAssistants', 'TooManyTestMailsTargetToPDTMachine', 'SingleMachineConnectivity', 'TooManyTestMailsTargetToPDTMachine', 'PerfTestSKU', 'HubPortExhaustion', 'HubPortExhaustion', 'SelfRecover', 'MissingAuthoritativeDomain', 'MissingAuthoritativeDomain', 'SelfRecover', 'TenantCustomizationEnabled', 'DLP-Flight', 'TenantCustomizationEnabled', 'TenantCustomizationEnabled', 'SafeAttachmentProcessingAgent', 'TenantCustomizationEnabled', 'CafeReturnWrongServerList', 'CliqueStampMissing_GriffinRegKeyMissing', 'Component-State', 'DLP-Flight', 'ItemAssistants', 'MDBReplication', 'MDBReplication', 'FHSSpareServerFail', 'jpnprd01_TYC_DC', 'PFA_agent_busy', 'Component-State', 'HubPortExhaustion', 'FhsSpareServerFail', 'BadADServerObjects', 'TenantCustomizationEnabled', 'DecommissionedAF', 'NetworkingSwitchedOutTooManySites', 'TooManyTestMailsTargetToPDTMachine', 'MdbReplication', 'CodeRegression_SDP', 'HubPortExhaustion', 'ADTypeInitializerException', 'SettingChangeEnabled', 'Component-State', 'SelfRecover', 'HubPortExhaustion', '404NotFound', 'MDBReplication', 'Auth_EmptyEPKClaim', 'HubPortExhaustion', 'UseRouteResolutionEnabled', 'SelfRecover', 'UseRouteResolutionEnabled', 'UseRouteResolutionEnabled', 'HubPortExhaustion', 'UseRouteResolutionEnabled', 'UseRouteResolutionEnabled', 'ADForestDiscoveryDelay', 'UseRouteResolutionEnabled', 'UseRouteResolutionEnabled', 'HubPortExhaustion', 'ADForestDiscoveryDelay', 'SelfRecover', 'UseRouteResolutionEnabled', 'UseRouteResolutionEnabled', 'ADForestDiscoveryDelay', 'Auth_EmptyEPKClaim', 'DecommissionedAF', 'MdbThrottling', 'DecommissionedAF', 'DecommissionedAF', 'UnhandledMimeExceptionInSmtp', 'per-source-limit', 'HubPortExhaustion', 'MdbThrottling', 'HubPortExhaustion', 'Component-State', 'XSOBug', 'NetworkingIssue', 'NetworkingIssue', 'Component-State', 'UnauthorizedAccessException', 'XSOBug', 'XSOBug', 'TooManyTestMailsTargetToPDTMachine', 'TooManyTestMailsTargetToPDTMachine', 'HubPortExhaustion', 'XSOBug', 'XSOBug', 'TooManyTestMailsTargetToPDTMachine', 'HubPortExhaustion', 'HubPortExhaustion', 'VIPs_Overloaded', 'Monitoring', 'SelfRecover', 'Probe-issue', 'RmiCoordinator', 'HubPortExhaustion', 'HubPortExhaustion', 'PerfTestSKU', 'PerfTestSKU', 'PerfTestSKU', 'CodeRegression_SDP', 'CodeRegression_SDP', 'CodeRegression_SDP', 'CodeRegression_SDP', 'LDAPSCertNotInstalled', 'CodeRegression_SDP', 'LDAPSCertNotInstalled', 'Auth_EmptyEPKClaim', 'Auth_EmptyEPKClaim', 'Auth_EmptyEPKClaim', 'HubPortExhaustion', 'HubPortExhaustion', 'HubPortExhaustion', 'MDBReplication', 'HubPortExhaustion', 'HubPortExhaustion', 'HubPortExhaustion', 'HubPortExhaustion', 'HubPortExhaustion', 'HubPortExhaustion', 'Auth_EmptyEPKClaim', 'HubPortExhaustion', 'ItemAssistants', 'HubPortExhaustion']
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

acc = accuracy_score(y_true, y_pred)
f1_micro = f1_score(y_true, y_pred, average='micro')
acc=accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
print('acc:',acc)
print('f1_micro:',f1_micro)
print('f1_macro:',f1_macro)