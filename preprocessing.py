import numpy as np
import pandas as pd
import re, nltk, spacy

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
tokenizer = RegexpTokenizer(r'\w+')

import gensim
from gensim.models import Phrases

import warnings
warnings.filterwarnings('ignore')

import sqlite3
db = "new_db.db"
con = sqlite3.connect(db)
cur = con.cursor()

# Query emails_main table for email_body and message_id
def query_emailbody():
    email_body = list(cur.execute("""SELECT (COALESCE(email_subject, '') || ' ' || 
                                     COALESCE(email_message, '')) 
                                     FROM emails_main WHERE folder_directory IS NULL"""))                                     

    email_body = [email[0] for email in email_body]
    return email_body

def query_messageid():
    email_messageid = list(cur.execute("""SELECT message_id FROM emails_main WHERE folder_directory IS NULL"""))
    email_messageid = [email[0] for email in email_messageid]
    return email_messageid

# Text Preprocessing
def preprocess_text(text):
    # Remove the common error messages
    for email in text:
        email.replace('Confidentiality Notice This message is being sent by or on behalf of a lawyer. It is intended exclusively for the individual or entity to which it is addressed. This communication may contain information that is proprietary, privileged or confidential or otherwise legally exempt from disclosure. If you are not the named addressee, you are not authorized to read, print, retain, copy or disseminate this message or any part of it. If you have received this message in error, please notify the sender immediately by e-mail and delete all copies of the message.', '')
        email.replace('Original Message', '')

    # Remove emails in email body as emails act as noise
    emails_data = []
    for i in range(len(text)):
        split_data = str(text[i]).split()
        emails_data.append(split_data)
        
    for i in range(len(emails_data)):
        for x in range(len(emails_data[i])):
            try:
                if '@' in emails_data[i][x]:
                    del emails_data[i][x]
            except IndexError:
                pass
            continue

    # Read in stopwords
    stopwords_list = []
    with open("stopwords_list.txt", "r") as fin:
        for line in fin.readlines():
            stopwords_list.append(line.strip())

    # Gensim simple preprocessing - convert list into lowercase and tokenize
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence)))        
    data_words = list(sent_to_words(emails_data))

    # Lemmatization
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'ADV'])
    data_tokenized = [tokenizer.tokenize(str(sentence)) for sentence in data_lemmatized]

    # Remove stopwords
    data_remove_sw = [[s for s in sentence if s not in stopwords_list] for sentence in data_tokenized]

    # Remove words that have less than 3 characters
    data_remove_2chars = [[s for s in sentence if len(s) > 2] for sentence in data_remove_sw]

    # Compute bigrams
    bigram = Phrases(data_remove_2chars, min_count=10)
    for idx in range(len(data_remove_2chars)):
        new_bigrams = []
        for token in bigram[data_remove_2chars[idx]]:
            if '_' in token:
                data_remove_2chars[idx].append(token)
                new_bigrams.append(token)
    documents = [' '.join([str(c) for c in lst]) for lst in data_remove_2chars]

    for i in range(len(documents)):
        new_bigrams = []
        
        for x in range(len(documents[i])):
            token = documents[i][x]
            if '_' in token:
                new_bigrams.append(token)         
    
        new_split_bigrams = []
        unique_bigrams = list(set(new_bigrams))
        for bigram in unique_bigrams:
            split_bi = bigram.split('_')
            first_term = split_bi[0]
            second_term = split_bi[1]
            new_split_bigrams.append([first_term, second_term])

        for doc in range(len(documents[i])-1):
            try:
                token = documents[i][doc]
                next_token = documents[i][doc+1]

                for a in range(len(new_split_bigrams)):
                    if token == new_split_bigrams[a][0] and next_token == new_split_bigrams[a][1]:
                        del documents[i][doc:doc+2]
            except IndexError:
                pass
            continue

    return documents    

# Insert preprocessed text to preprocessed_emails table for later queries
def insert_preprocessed_text(preprocessed_values):
    
    try:
        con.executemany("INSERT INTO preprocessed_emails(message_id, preprocessed_text) VALUES (?, ?)", preprocessed_values)

    except sqlite3.IntegrityError:
        pass    

    con.commit()
        
def main():
    preprocessed_text = preprocess_text(query_emailbody())
    
    preprocessed_values = list(zip(query_messageid(), preprocessed_text))
    print(preprocessed_values[0])
    insert_preprocessed_text(preprocessed_values)

if __name__ == '__main__':
    main()