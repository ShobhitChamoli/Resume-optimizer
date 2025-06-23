import pandas as pd
import numpy as np
import joblib
import os
import re
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_resume(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", ' ', text)
    text = re.sub(r'\@\w+|\#', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Load and clean dataset
df = pd.read_csv("C:\\Users\\ACER\\Desktop\\resume_dataset_job_skills.csv", encoding="utf-8")
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Encode labels
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# Vectorize text
tfidf = TfidfVectorizer(max_features=1500, stop_words='english')
X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category_encoded']

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(le, "models/label_encoder.joblib")

category_vectors = {cat: X[y == cat] for cat in np.unique(y)}
joblib.dump(category_vectors, "models/category_vectors.joblib")

print("Computation completed")