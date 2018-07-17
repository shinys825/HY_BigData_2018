import json
import re
import pandas as pd
import numpy as np
from gensim.models import doc2vec
from scipy import spatial

def get_label_sims(input_doc, model):
    review = input_doc[0].split(' ')
    inferred_vector = model.infer_vector(review, steps=20, alpha=0.025)

    good_vec = model.docvecs['good']
    bad_vec = model.docvecs['bad']

    good_sim = 1 - spatial.distance.cosine(inferred_vector, good_vec)
    bad_sim = 1 - spatial.distance.cosine(inferred_vector, bad_vec)
    return good_sim, bad_sim


def get_accuracy(test_reviews, train_reviews, model):
    correct_sentiment = 0
    for i in range(len(test_reviews)):
        doc = test_reviews[i]
        sentiment = doc[2]
        similar_docs = get_similar_doc(test_reviews[i], train_reviews, model, verbose=False)
        tag, cnt = np.unique(similar_docs['sentiment'], return_counts=True)
        predicted_sentiment = tag[np.argmax(cnt)]
        if sentiment == predicted_sentiment: correct_sentiment += 1
    return correct_sentiment

def make_doc2vec_object(window=5, vector_size=50, min_count=3):
    return doc2vec.Doc2Vec(window=window, vector_size=vector_size, alpha=0.1, min_alpha=0.01, dm=0,
                           sample=1e-3, hs=0, negative=5, min_count=min_count, workers=-1)

def get_similar_doc(input_doc, train_docs, model, verbose=True):
    review = input_doc[0].split(' ')
    inferred_vector = model.infer_vector(review, steps=20, alpha=0.025)
    similar_result = [x for x in model.docvecs.most_similar(positive=[inferred_vector], topn=5)]
    similar_score = [float('%4f'%score[1]) for score in similar_result]
    similar_indices = [idx[0][1:] for idx in similar_result]

    if verbose:
        print('Review: \n', input_doc[0], '\n')
        print('Rating:', input_doc[1])
        print('Sentiment:', input_doc[2])
        print('Category:', input_doc[3])
        
    
    results = list()
    for i in range(len(similar_indices)):
        idx = int(similar_indices[i])
        doc = train_docs[idx][0]
        rating = train_docs[idx][1]
        sentiment = train_docs[idx][2]
        category = train_docs[idx][3]
        score = similar_score[i]
        result = (idx, doc, rating, sentiment, category, score)
        results.append(result)
    results_df = pd.DataFrame(results, columns=['index', 'review', 'rating', 'sentiment', 'category', 'score'])
    return results_df

def prepare_reviews():
    with open('./Automotive_5.json', 'r') as f:
        auto = f.readlines()
    with open('./Musical_Instruments_5.json', 'r') as f:
        music = f.readlines()
    with open('./Patio_Lawn_and_Garden_5.json', 'r') as f:
        garden = f.readlines()
        
    auto_review = _get_review_list(auto, 'auto')
    music_review = _get_review_list(music, 'music')
    garden_review = _get_review_list(garden, 'garden')
    
    train_auto = auto_review[:-100]
    train_music = music_review[:-100]
    train_garden = garden_review[:-100]
    train_reviews = train_auto + train_music + train_garden
    
    test_auto = auto_review[-100:]
    test_music = music_review[-100:]
    test_garden = garden_review[-100:]
    test_reviews = test_auto + test_music + test_garden
    return train_reviews, test_reviews

def _get_review_list(review_dict, category):
    text_pattern = re.compile(r'[a-zA-Z]{2,}')
    reviews = list()
    for txt in review_dict:
        review = json.loads(txt)['reviewText']
        review = ' '.join(text_pattern.findall(review))
        review = review.lower()
        
        rating = json.loads(txt)['overall']
        if rating < 3: sentiment = 'bad'
        if rating >= 4: sentiment = 'good'
        if rating == 3: continue
        review_tuple = (review, rating, sentiment, category)
        reviews.append(review_tuple)
    return reviews

def doc2vec_labeler(docs):
    labeled_docs = list()
    for i in range(len(docs)):
        doc = docs[i]
        words = doc[0].split(' ')
        sentiment = doc[2]
        category = doc[3]
        labels = ['d'+str(i), str(sentiment)]
        sentence = doc2vec.TaggedDocument(words=words, tags=labels)
        labeled_docs.append(sentence)
    return labeled_docs

