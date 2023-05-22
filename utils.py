import fasttext
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')


import re
from nltk.corpus import stopwords
import cufflinks as cf
from transformers import AutoTokenizer
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
def text_clean(text):# used for fasttext
    cache_english_stopwords = stopwords.words('english')
    cache_english_stopwords+=['!', ',', '.', '?', '-s', '-ly', '</s> ', 's','[',']',':','(',')','{','}','\'','<','>','+','-','_','__','--','|','\'\'']
    # Remove HTML tags (e.g. &amp;)
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', text)
    # Remove certain value symbols
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    # Remove hyperlinks
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    # # Remove some specialized abbreviation words, in other words, words with fewer letters
    string4 = " ".join(re.findall(r'\b\w+\b', text_no_hyperlinks))
    # Tokenization
    tokens = word_tokenize(string4)
    # Remove stopwords
    list_no_stopwords = [i for i in tokens if i not in cache_english_stopwords]
    # Final filtered result
    text_filtered = ' '.join(list_no_stopwords)  # ''.join() would join without spaces between words.
    return text_filtered
def count_word_num(text):
    if text is np.nan:
        return 0
    else:
        return len(text.split())

def truncate_text(text, length):
    if text is np.nan:
        return text
    else:
        return ' '.join(text.split()[:length])

def trainingFile_preprocess(df,test=False):

    label_list=df['Keyword'].apply(lambda x:'-'.join(x.split()))
    document_list=df['metadata']
    text_filtered_list=document_list.apply(lambda x:text_clean(str(x['KeyValuePairs'])+' '+x['AlertType']+' '+str(x['ScopeType'])+' '+x['AdviceDetail']))  #str(x['KeyValuePairs'])+' '+
    if not test:
        file=np.array('__label__'+label_list+' '+text_filtered_list)
    else:
        file=np.array(text_filtered_list)
    return file
def draw_confusion_matrix(y_true, y_pred, class_names, figsize):
    print("There are {} unique RootCauseCategories".format(len(class_names)))

    le = LabelEncoder()
    le.fit(class_names)  # class_names is a serious superset of both y_true and y_pred
    y_true_num = le.transform(y_true)
    y_pred_num = le.transform(y_pred)

    conf_mat = confusion_matrix(y_true_num, y_pred_num, normalize='true')
    conf_mat_text = pd.DataFrame(conf_mat, columns=le.classes_, index=le.classes_)

    class_counts = np.bincount(y_true_num)
    sort_indices = np.argsort(class_counts)[::-1]  # reverse the sort
    conf_mat_reordered = conf_mat_text.reindex(index=le.classes_[sort_indices], columns=le.classes_[sort_indices])
    # change column names
    conf_mat_reordered.columns = [le.classes_[sort_indices][i].split(":")[0] for i in
                                  range(len(le.classes_[sort_indices]))]

    plt.figure(figsize=figsize, dpi=100)
    ax1 = sns.heatmap(conf_mat_reordered, annot=True, fmt='.2f', cmap='Blues', square=True, cbar=False)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # when exporting as jpg, use the command below...
    plt.show()

def clean(text,max_len=700):#used for GPT
    content = text
    content = re.sub(r'\.', ' ', content)
    content = re.sub(r',', ' ', content)
    # Replace all whitespaces
    content = content.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\\\', ' ')
    content = content.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\\', ' ')
    # Remove all quotes and brackets
    content = content.replace('\"', ' ').replace("\'", ' ').replace('(', ' ').replace(')', ' ')
    # Remove all vertical bars
    content = content.replace('|', ' ')
    # Replace all consecutive '-'s with only one '-'
    content = re.sub('-+', '-', content)
    # Remove filenames if its extension is in 'file_exts.txt'
    common_file_extensions = open('Data_Models/file_exts.txt', 'r').read().splitlines()
    content = ' '.join([word for word in content.split() if '.' + word.split('.')[-1] not in common_file_extensions])
    # If there are multiple whitespaces, replace with one whitespace
    content =  re.sub(' +', ' ', content)
    content = re.sub(r"http\S+", '', content)
    content = re.sub(r'\$\w*', '', content)
    content = re.sub(r'\w{8}-\w{4}-\w{4}-\w{4}-\w{12}', ' ', content)
    content = truncate_text(content, max_len) if count_word_num(content) > max_len else content
    return content

def process(text):#handle GPT summmary response
    match = re.search(r'Summary:(.*?)\n|Output:(.*?)\n\n', text, re.DOTALL)
    output_text=''
    if match:
        summary_text = match.group(1)
        if summary_text:
            output_text=summary_text
        else:
            output_text=match.group(2)
    else:
        return text
    if not output_text:
        match=re.search(r'Summary:\n(.*?)\n',text,re.DOTALL)
        if match:
            output_text=match.group(1)
        else:output_text=' '
    output_text=output_text.replace('\n',' ')
    if output_text==' ' or output_text=='':
        return text
    return output_text