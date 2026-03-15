import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec = pickle.load(open("vectorizer.pkl","rb"))
my_question_vec_data = pickle.load(open("question_vectors.pkl","rb"))
my_faq_data = pickle.load(open("faq_data.pkl","rb"))

questions = list(my_faq_data.keys())

st.title("QA Bot")

user_question = st.text_input("Ask question")

if st.button("Get Answer"):
    
    q = vec.transform([user_question])
    sim = cosine_similarity(q, my_question_vec_data).flatten()
    
    idx = np.argmax(sim)
    
    answer = my_faq_data[questions[idx]]
    
    st.write(answer)