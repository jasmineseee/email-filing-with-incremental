import numpy as np
import pandas as pd
import re, nltk, spacy

import pyLDAvis
import pyLDAvis.sklearn

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
tokenizer = RegexpTokenizer(r'\w+')

import gensim
from gensim.models import ldamodel, Phrases

import warnings
warnings.filterwarnings('ignore')

import preprocessing

import sqlite3
db = "new_db.db"
con = sqlite3.connect(db)
cur = con.cursor()

def unsupervised_learning(preprocessed_text, documents_to_tokens, message_id):
    # Apply term weighting with TF-IDF (Term Frequency-Inverse Document Frequency)
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df=10)
    # tf-idf matrix of documents
    try:
        tfidf = tfidf_vectorizer.fit_transform(preprocessed_text)
    except ValueError:
        pass
    # Extract the resulting vocabulary
    terms = tfidf_vectorizer.get_feature_names()

    # NMF Unsupervised Learning
    # Create the Topic Models
    # We need to start by pre-specifying an initial range of "sensible" values then apply NMF for each of these values.
    from sklearn.decomposition import NMF

    kmin, kmax = 4, 10

    topic_models = []
    for k in range(kmin, kmax+1):
        # Run NMF
        model = NMF(init="nndsvd", n_components=k) 
        W = model.fit_transform(tfidf)
        H = model.components_    
        # Store for later
        topic_models.append((k,W,H))

    # Build a Word Embedding
    # To select the optimal number of topics, a topic coherence measure called TC-W2V is used. 
    # This measure relies on the use of a word embedding model constructed from the corpus.
    # TC-W2V uses the skip-gram algorithm, which predicts the context words based on the current word, 
    # for estimating word representations in a vector space.
    w2v_model = gensim.models.Word2Vec(documents_to_tokens, size=300, min_count=10, sg=1)

    # Calculate Coherence
    # The coherence score is the mean pairwise Cosine similarity of two term vectors generated with the skip-gram model.
    from itertools import combinations
    def calculate_coherence(w2v_model, term_rankings):
        overall_coherence = 0.0
        for topic_index in range(len(term_rankings)):
            # Check each pair of terms
            pair_scores = []
            for pair in combinations(term_rankings[topic_index], 2):
                pair_scores.append(w2v_model.similarity(pair[0], pair[1]))
            # Get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            overall_coherence += topic_score
        # Get the mean score across all topics
        return overall_coherence / len(term_rankings)

    # Topic Descriptors
    # The top ranked terms from the H factor for each topic can give us an insight into the content of that topic.
    def get_descriptor(all_terms, H, topic_index, top):
        # Reverse sort the values to sort the indices
        top_indices = np.argsort(H[topic_index,:])[::-1]
        # Now get the terms corresponding to the top-ranked indices
        top_terms = []
        for term_index in top_indices[0:top]:
            top_terms.append(all_terms[term_index])
        return top_terms

    # The higher the coherence, the more human interpretable the topic is.
    k_values = []
    coherences = []
    for (k,W,H) in topic_models:
        # Get all of the topic descriptors - the term_rankings, based on top 10 terms
        term_rankings = []
        for topic_index in range(k):
            term_rankings.append(get_descriptor(terms, H, topic_index, 10))
        # Now calculate the coherence based on our Word2vec model
        k_values.append(k)
        coherences.append(calculate_coherence(w2v_model, term_rankings))
        
    # Topic Evaluation: Automated Selection of Important Topics
    # best_k is the optimal number of topics
    best_k = coherences.index(max(coherences)) + kmin
    # Get the model that we generated earlier.
    W = topic_models[best_k-kmin][1]
    H = topic_models[best_k-kmin][2]

    # Now that we know best_k is the optimal number of topics, we want to find out which topic(s) in best_k number of topics have high coherence. 
    # Similar to the coherence measure above.
    def calculate_coherence_topic(w2v_model, term_rankings):
        coherence_of_topics = []
        
        for topic_index in range(len(term_rankings)):
            # Check each pair of terms
            pair_scores = []
            for pair in combinations(term_rankings[topic_index], 2):
                pair_scores.append(w2v_model.similarity(pair[0], pair[1]))
            # Get the mean for all pairs in this topic
            topic_score = sum(pair_scores) / len(pair_scores)
            coherence_of_topics.append(topic_score)
            
        return coherence_of_topics

    # Create a list of topic numbers
    topic_index_list = []
    for i in range(1, best_k+1):
        topic_index_list.append(i)

    # Get all of the topic descriptors - the term_rankings, based on top 10 terms 
    # and calculate coherence for each topic within best_k
    term_rankings = []
    for topic_index in range(best_k):
        term_rankings.append(get_descriptor(terms, H, topic_index, 10))
        
    coherences_scores_list = calculate_coherence_topic(w2v_model, term_rankings)

    # Get the name of the topic
    # The name of each topic will be the word with highest relavance to respective topic
    topic_name_list = []
    for topic_index in range(best_k):
        descriptor = get_descriptor(terms, H, topic_index, 1)
        topic_name_list.append(descriptor)

    # Token percentage can be used as an additional measure to weed out irrelevant topics. 
    # If the token percentage is too low despite having high coherence, it is not a reliable enough topic.
    # pyLDAvis - To get Token Percentage
    final_nmf = NMF(init="nndsvd", n_components=best_k).fit(tfidf)
    p = pyLDAvis.sklearn.prepare(final_nmf, tfidf, tfidf_vectorizer)
    token_percentage_list = (p.topic_coordinates['Freq']).tolist()

    # A dictionary of the topic number and topic name
    topic_reference_dict = dict(zip(topic_index_list, topic_name_list))

    finaldf = pd.DataFrame({
        'Topic': topic_index_list,
        'Coherence Score': coherences_scores_list,
        'Token Percentage': token_percentage_list
    })

    good_coherence_df = finaldf.loc[(finaldf['Coherence Score'] >= 0.80) & (finaldf['Token Percentage'] >= 10.0)]
    good_coherence_topics = good_coherence_df['Topic'].tolist()

    nmf_topic_membership_weights = (final_nmf.transform(tfidf)).argmax(axis=1).tolist()
    nmf_topics = []

    for i in nmf_topic_membership_weights:
        nmf_topics.append(i+1)

    final_df = pd.DataFrame({'Topic': nmf_topics, 'message_id': message_id})
    final_df = final_df[final_df['Topic'].isin(good_coherence_topics)]

    # Replace topic number with topic name to make it even more interpretable for end users
    final_df['Topic'] = final_df['Topic'].replace(topic_reference_dict)

    return final_df

def main():
    preprocessed_text = preprocessing.preprocess_text(preprocessing.query_emailbody())
    documents_to_tokens = [tokenizer.tokenize(sentence) for sentence in preprocessed_text]
    message_id = preprocessing.query_messageid()
    final_df = unsupervised_learning(preprocessed_text, documents_to_tokens, message_id)

    # Create a temporary table to store the results of unsupervised learning
    final_df.to_sql('unsupervised_temp', con, if_exists='replace')

    # Update folder_directory in emails table and delete temporary table for unsupervised learning
    cur.executescript("""UPDATE emails_main
                         SET folder_directory = (
                            SELECT Topic 
                            FROM unsupervised_temp 
                            WHERE message_id = emails_main.message_id)
                         WHERE emails_main.folder_directory IS NULL;

                         DROP TABLE unsupervised_temp;""")
    con.commit()

if __name__ == '__main__':
    main()

