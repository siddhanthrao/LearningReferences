
# coding: utf-8

# In[4]:

## output path for csv files
path="C:/Users/msingh42.MS/Documents"


# In[5]:

# Reading data for analysis
text1 = pd.read_csv(r'C:/Users/msingh42.MS/Documents/WC_Data_Science.csv', sep=',', encoding='iso-8859-1')


# In[6]:

text1.dropna(axis=0,how='any',inplace=True)


# In[7]:

text=text1["Text"].values.tolist()


# In[6]:

##Topic modelling using LDA


# In[104]:

text[0]


# In[105]:

def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in stopwords and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text


# In[ ]:

ss=clean_text(text[0])


# In[ ]:

import time
t1=time.time()


# In[8]:

a=0
NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')
tweet_text=text  # text is a list of column to be modelled
doc_complete=tweet_text
#doc_complete_2=[]
doc_complete_2=tweet_text
'''for i in range(len(doc_complete)):
    try:
        doc_complete1=doc_complete[i].encode('utf-8').strip()
        doc_complete_2.append(doc_complete1)
    except:
        continue''' 
def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    return cleaned_text 
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for w in doc_complete_2:
    a=a+1
    tokenized_data.append(clean_text(w))


# In[ ]:




# In[110]:

a=0
NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')
tweet_text=text  # text is a list of column to be modelled
doc_complete=tweet_text
#doc_complete_2=[]
doc_complete_2=tweet_text
custom_list=["fall"]
'''for i in range(len(doc_complete)):
    try:
        doc_complete1=doc_complete[i].encode('utf-8').strip()
        doc_complete_2.append(doc_complete1)
    except:
        continue''' 
def clean_text(text,custom_list):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    cleaned_text1 = [t for t in cleaned_text if t not in custom_list]
    return cleaned_text 
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for w in doc_complete_2:
    a=a+1
    tokenized_data.append(clean_text(w,custom_list))
# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)
# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data] 
# Have a look at how the Nth document looks like: [(word_id, count), ...]
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...
 # Build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
final=[]
for topic in lda_model.show_topics(num_topics=NUM_TOPICS, formatted=False, num_words=6):
    topicwords = [w for (w, val) in topic[1]]
    topicwords_val = [val for (w, val) in topic[1]]
    final.append([topicwords,topicwords_val])
final1=pd.DataFrame(final,columns=["topic","prob"])


# In[54]:

path


# In[111]:


final1.to_csv(path+"/topics.csv")


# In[26]:

t2=time.time()
t2-t1


# In[ ]:




# n gram analysis

# In[112]:

import re
import string
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
def data_cleaning(tweet,custom_list):
    tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
    tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
    #stop = set(stopwords.words('english'))
    tweet = re.sub(r'[^a-zA-Z0-9]'," ",tweet)
    stop_words=set(['a',    'about', 'above', 'after',   'again',  'against',              'ain',      'all',        'am',               'an',       'and',     'any',     'are',      'as',        'at',        'be',       'because',            'been',   'before',               'being',  'below', 'between',           'both',   'but',      'by',        'can',     'couldn',               'd',               'did',      'didn',    'do',       'does',   'doesn', 'doing',  'don',     'down',  'during',               'each',               'few',     'for',      'from',   'further',              'had',     'hadn',   'has',      'hasn',   'have',   'haven',               'having',               'he',       'her',      'here',    'hers',    'herself',              'him',     'himself',               'his',       'how',    'i',           'if',         'in',         'into',     'is',         'isn',       'it',         'its',        'itself',               'just',     'll',          'm',        'ma',      'me',      'mightn',              'more',  'most',   'mustn', 'my',               'myself',               'needn', 'now',    'o',         'of',        'off',      'on',       'once',   'only',    'or',               'other',  'our',      'ours',    'ourselves',          'out',      'over',    'own',    're',        's',          'same',               'shan',   'she',      'should',               'so',        'some',  'such',    't',          'than',    'that',    'the',               'their',   'theirs',  'them',  'themselves',      'then',    'there',  'these',  'they',    'this',     'those',               'through',            'to',        'too',     'under', 'until',    'up',       've',        'very',    'was',     'we',               'were',   'weren',               'what',   'when',  'where',               'which', 'while',  'who',    'whom',               'why',    'will',      'with',    'won',    'y',          'you',     'your',    'yours',  'yourself',               'yourselves'])
    exclude = set(string.punctuation)
    exclude1= set(custom_list)
    stop_words.update(exclude1)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in tweet.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


# In[113]:

from nltk import ngrams


# In[44]:




# In[114]:

def cal_ngram(text,n):
    token = nltk.word_tokenize(text)
    n_grams = ngrams(token,n)
    return n_grams


# In[115]:

n_gram = cal_ngram(text[0],2)   # generating N grams for input data


# In[64]:

type(text)


# In[65]:

type(text[0])


# In[116]:

w = ''
for v in text:
    w+=" "+v


# In[117]:

abc = data_cleaning(w,["I","me","my","myself","we","us","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","whose","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","will","would","should","can","could","ought","i'm","you're","he's","she's","it's","we're","they're","i've","you've","we've","they've","i'd","you'd","he'd","she'd","we'd","they'd","i'll","you'll","he'll","she'll","we'll","they'll","isn't","aren't","wasn't","weren't","hasn't","haven't","hadn't","doesn't","don't","didn't","won't","wouldn't","shan't","shouldn't","can't","cannot","couldn't","mustn't","let's","that's","who's","what's","here's","there's","when's","where's","why's","how's","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","upon","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","say","says","said","shall"])


# In[57]:

w[0:100]


# In[ ]:




# In[118]:

n_gram = cal_ngram(abc,2)   # generating N grams for input data
n_gram_common = Counter(n_gram).most_common()  # Listing top 10 trending N grams
n_gram_df=pd.DataFrame(n_gram_common,columns=["N Gram","Frquency"])


# In[119]:

n_gram_df


# ##TF_IDF

# In[120]:

n_gram_df.shape, text1.shape


# In[145]:

sample = n_gram_df.loc[:100]


# In[ ]:




# In[82]:

text1.columns


# In[83]:

keyword_1 = "lower"
keyword_2 = "back"


# In[146]:

def extract_claim_cost(df):
    #print(df)
    text = df['Text']
    if (keyword_1 in text) | (keyword_2 in text):
        return df['Claim Cost']
    else:
        return 0


# In[147]:

sum_list = []
for idx in tqdm(range(sample.shape[0])):
    one = sample.iloc[idx,0][0]
    two = sample.iloc[idx,0][1]
    
    keyword_1 = one
    keyword_2 = two
    sum_ = text1.apply(extract_claim_cost, axis=1)
    sum_list.append(sum_.sum())


# In[133]:

from tqdm import tqdm


# In[ ]:




# In[ ]:




# In[148]:

sum_ = text1.apply(extract_claim_cost, axis=1)


# In[149]:

sum_.sum()


# In[95]:

text1['Text'] = text1['Text'].apply(lambda x : str(x).lower())


# In[96]:

text1['Text'].head()


# In[150]:

sum_list


# In[151]:

res = pd.concat([sample, pd.Series(sum_list, name='cost')], axis=1).reset_index(drop=True)


# In[154]:

res.to_csv(path+"/N_Gram.csv")


# In[ ]:



