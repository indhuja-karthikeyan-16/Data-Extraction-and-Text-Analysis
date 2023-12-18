
#importing library
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from google.colab import files
import nltk
import string
nltk.download('punkt')



#importing input file
df=pd.read_csv('/content/Input.csv')[['URL_ID','URL']]

df=df.iloc[0:150]

df

df.drop('URL_ID',axis=1,inplace=True)

"""# Data extraction"""

#extracting text from all the url
url_id=1
for i in range(0,len(df)):

   j=df.iloc[i].values

   headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}#giving user access
   page=requests.get(j[0],headers=headers)
   soup=BeautifulSoup(page.content,'html.parser')
   content=soup.findAll(attrs={'class':'td-post-content'})
   content=content[0].text.replace('\xa0',"  ").replace('\n',"  ")
   title=soup.findAll(attrs={'class':'entry-title'})
   title=title[16].text.replace('\n',"  ").replace('/',"")
   text=title+ '.' +content
   text=np.array(text)
   text.reshape(1,-1)
   df1=pd.Series(text)
   b=str(url_id)+"."+'txt'
   with open(b, 'a') as f:
    df1.to_csv(f, line_terminator=',', index=False, header=False)
   files.download(b)
   url_id+=1

"""# Data analysis"""

#importing each extracted files
text=pd.read_csv("/content/100.txt",header=None)

#information of data frame
text.info()

#removing extra created column
text.drop(1,axis=1,inplace=True)

#converting type
text=text.astype(str)

#converting text to sentence
import re
a=text[0].str.split('([\.]\s)',expand=False)
b=a.explode()
b=pd.DataFrame(b)
b.columns=['abc']

#removing . char from each rows
def abcd(x):
    nopunc =[char for char in x if char != '.']
    return ''.join(nopunc)
b['abc']=b['abc'].apply(abcd)

#replacing emty space with null values
c=b.replace('',np.nan,regex=True)
c=c.mask(c==" ")
c=c.dropna()
c.reset_index(drop=True,inplace=True)

c

#importing nltk library and stopwords
import nltk
import string

punc=[punc for punc in string.punctuation]

punc

#importing stop words files that are provided
StopWords_Auditor=pd.read_csv("/content/StopWords_Auditor.txt",header=None)
StopWords_Currencies = pd.read_csv("/content/StopWords_Currencies.txt", header=None, encoding="ISO-8859-1",sep='\n')
StopWords_DatesandNumbers=pd.read_csv("/content/StopWords_DatesandNumbers.txt",header=None)
StopWords_Generic=pd.read_csv("/content/StopWords_Generic.txt",header=None)
StopWords_GenericLong=pd.read_csv("/content/StopWords_GenericLong.txt",header=None)
StopWords_Geographic=pd.read_csv("/content/StopWords_Geographic.txt",header=None)
StopWords_Names=pd.read_csv("/content/StopWords_Names.txt",header=None)

#creating func for removing stop words
def text_process(text):
    nopunc =[char for char in text if char not in punc or char not in [':',',','(',')','’','?']]
    nopunc=''.join(nopunc)
    txt=' '.join([word for word in nopunc.split() if word.lower() not in StopWords_Auditor])
    txt1=' '.join([word for word in txt.split() if word.lower() not in StopWords_Currencies])
    txt2=' '.join([word for word in txt1.split() if word.lower() not in StopWords_DatesandNumbers])
    txt3=' '.join([word for word in txt2.split() if word.lower() not in StopWords_Generic])
    txt4=' '.join([word for word in txt3.split() if word.lower() not in StopWords_GenericLong])
    txt5=' '.join([word for word in txt4.split() if word.lower() not in StopWords_Geographic])
    return ' '.join([word for word in txt5.split() if word.lower() not in StopWords_Names])

#applying func for each row
nopunc

c

#importing master Dictionary
positive=pd.read_csv("/content/positive-words.txt",header=None)
negative=pd.read_csv("/content/negative-words.txt",header=None,encoding="ISO-8859-1",sep='\n' )

positive.columns=['abc']
negative.columns=['abc']
positive['abc']=positive['abc'].astype(str)
negative['abc']=negative['abc'].astype(str)

#positive and negative dictionary without stopwords
positive['abc']=positive['abc'].apply(text_process)
negative['abc']=negative['abc'].apply(text_process)

#positive list
length=positive.shape[0]
post=[]
for i in range(0,length):
   nopunc =[char for char in positive.iloc[i] if char not in string.punctuation or char != '+']
   nopunc=''.join(nopunc)

   post.append(nopunc)

#negative list
length=negative.shape[0]
neg=[]
for i in range(0,length):
  nopunc =[char for char in negative.iloc[i] if char not in string.punctuation or char != '+']
  nopunc=''.join(nopunc)
  neg.append(nopunc)

#importing tokenize library
from nltk.tokenize import word_tokenize

txt_list=[]
length=c.shape[0]
for i in range(0,length):
  txt=' '.join([word for word in c.iloc[i]])
  txt_list.append(txt)

#tokenization of text
tokenize_text=[]
for i in txt_list:

  tokenize_text+=(word_tokenize(i))

print(tokenize_text)

len(tokenize_text)

"""### 1) POSITIVE SCORE"""

positive_score=0
for i in tokenize_text:
  if(i.lower() in post):
    positive_score+=1
print('postive score=', positive_score)

"""### 2) NEGATIVE SCORE"""

negative_score=0
for i in tokenize_text:
  if(i.lower() in neg):
    negative_score+=1
print('negative score=', negative_score)

"""### 3) POLARITY SCORE"""

Polarity_Score=(positive_score-negative_score)/((positive_score+negative_score)+0.000001)
print('polarity_score=', Polarity_Score)

"""### 4) SUBJECTIVITY SCORE"""

subjectiivity_score=(positive_score-negative_score)/((len(tokenize_text))+ 0.000001)
print('subjectivity_score',subjectiivity_score)

"""### 5) AVG SENTENCE LENGTH"""

length=c.shape[0]
avg_length=[]
for i in range(0,length):
  avg_length.append(len(c['abc'].iloc[i]))
avg_senetence_length=sum(avg_length)/len(avg_length)
print('avg sentence length=', avg_senetence_length)

"""### 6) PERCENTAGE OF COMPLEX WORDS"""

vowels=['a','e','i','o','u']
import re
count=0
complex_Word_Count=0
for i in tokenize_text:
  x=re.compile('[es|ed]$')
  if x.match(i.lower()):
   count+=0
  else:
    for j in i:
      if(j.lower() in vowels ):
        count+=1
  if(count>2):
   complex_Word_Count+=1
  count=0

Percentage_of_Complex_words=complex_Word_Count/len(tokenize_text)
print('percentag of complex words= ',Percentage_of_Complex_words)

"""### 7) FOG INDEX"""

Fog_Index = 0.4 * (avg_senetence_length + Percentage_of_Complex_words)
print('fog index= ',Fog_Index )

"""### 8) AVG NUMBER OF WORDS PER SENTENCE"""

length=c.shape[0]
avg_length=[]
for i in range(0,length):
  a=[word.split( ) for word in c.iloc[i]]
  avg_length.append(len(a[0]))
  a=0
#avg
avg_no_of_words_per_sentence=sum(avg_length)/length
print("avg no of words per sentence= ",avg_no_of_words_per_sentence)

"""### 9) COMPLEX WORD COUNT"""

vowels=['a','e','i','o','u']
import re
count=0
complex_Word_Count=0
for i in tokenize_text:
  x=re.compile('[es|ed]$')
  if x.match(i.lower()):
   count+=0
  else:
    for j in i:
      if(j.lower() in vowels ):
        count+=1
  if(count>2):
   complex_Word_Count+=1
  count=0
print('complex words count=',  complex_Word_Count)

"""### 10) WORD COUNT"""

word_count=len(tokenize_text)
print('word count= ', word_count)

"""### 11) SYLLABLE PER WORD"""

vowels=['a','e','i','o','u']
import re
count=0
for i in tokenize_text:
  x=re.compile('[es|ed]$')
  if x.match(i.lower()):
   count+=0
  else:
    for j in i:
      if(j.lower() in vowels ):
        count+=1
syllable_count=count
print('syllable_per_word= ',syllable_count)

"""### 12) PERSONAL PRONOUNS"""

pronouns=['i','we','my','ours','us' ]
import re
count=0
for i in tokenize_text:
  if i.lower() in pronouns:
   count+=1
personal_pronouns=count
print('personal pronouns= ',personal_pronouns )

"""### 13) AVG WORD LENGTH"""

count=0
for i in tokenize_text:
  for j in i:
    count+=1
avg_word_length=count/len(tokenize_text)
print('avg word= ', avg_word_length)

data={'positive_score':positive_score,'negative_score':negative_score,'Polarity_Score':Polarity_Score,'subjectiivity_score':subjectiivity_score,'avg_senetence_length':avg_senetence_length,'Percentage_of_Complex_words':Percentage_of_Complex_words,'Fog_Index':Fog_Index,'avg_no_of_words_per_sentence':avg_no_of_words_per_sentence,'complex_Word_Count':complex_Word_Count,'word_count':word_count,'syllable_count':syllable_count,'personal_pronouns':personal_pronouns,'avg_word_length':avg_word_length}

output=pd.DataFrame()
output=output.append(data,ignore_index=True)
output.columns=['positive_score','negative_score','Polarity_Score','subjectiivity_score','avg_senetence_length','Percentage_of_Complex_words','Fog_Index','avg_no_of_words_per_sentence','complex_Word_Count','word_count','syllable_count','personal_pronouns','avg_word_length']
output

with open('output.csv', 'a') as f:#creating text file
    output.to_csv(f, index=False, header=False)
files.download('output.csv')
