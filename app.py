import re
import pickle
import numpy as np
import streamlit as st
from nltk.stem import PorterStemmer

stopwords = [
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'of', 'at', 'in', 
    'on', 'to', 'for', 'with', 'by', 'from', 'about', 'as', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
    'did', 'will', 'would', 'can', 'could', 'should', 'shall', 'may', 'might', 
    'must', 'that', 'this', 'these', 'those', 'it', 'its', 'he', 'she', 'his', 
    'her', 'they', 'them', 'their', 'you', 'your', 'we', 'us', 'our', 'i', 'me', 
    'my', 'mine', 'myself', 'yourself', 'himself', 'herself', 'itself', 'themselves', 
    'which', 'what', 'who', 'whom', 'when', 'where', 'why', 'how', 'up', 'down', 
    'out', 'into', 'under', 'over', 'more', 'so', 'very', 'too', 
    'just', 'than', 'any', 'much', 'many', 'some', 'other', 'such', 'each', 
    'every', 'all', 'only', 'again', 'ever', 'never', 'always', 'often', 
    'sometimes', 'less', 'most', 'few', 'several', 'own', 'same', 'both', 'either', 'movie'
    'film'
]

stemmer = PorterStemmer()

def clean_review_function(review: str) -> str:
    review = review.lower()
    review = review.replace('<br />', '')
    review = re.sub(r'[^\w\s]', '', review)
    review = re.sub(r'\d+', '', review)
    
    words = review.split(' ')
    clean_words = [stemmer.stem(word) for word in words if word not in stopwords]
    
    return clean_words

@st.cache_resource
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_pca():
    with open('pca_model.pkl', 'rb') as file:
        return pickle.load(file)

@st.cache_resource
def load_naive_bayes():
    with open('naive_bayes.pkl', 'rb') as file:
        return pickle.load(file)


st.markdown(
    """
    <h1 style="text-align: center;">📝 Sentiment Analysis App</h1>
    """, 
    unsafe_allow_html=True
)

input_review = st.text_input("Input a Review: ")
#submit = st.button("Analyze")

if input_review:
    cleaned_review = clean_review_function(input_review)
    
    vectorizer = load_vectorizer()
    pca_model = load_pca()
    naive_bayes = load_naive_bayes()

    vectorized_review = vectorizer.transform([' '.join(cleaned_review)]) 
    reduced_review = pca_model.transform(vectorized_review)

    prediction = naive_bayes.predict(reduced_review.reshape(1, -1))

    if prediction[0] == 0:
        st.markdown(
            """
            <div style="background-color:#f8d7da;padding:10px;border-radius:5px;text-align:center">
                <h3 style="color:#721c24;">😔 The review is NEGATIVE</h2>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#d4edda;padding:10px;border-radius:5px;text-align:center;">
                <h3 style="color:#155724;">😊 The review is POSITIVE</h2>
            </div>
            """, unsafe_allow_html=True
        )
