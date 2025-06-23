# 🧠 Resume Optimizer & Scorer

This project is a smart resume scoring and optimization tool built using **NLP**, **spaCy**, **Sentence Transformers**, and **Streamlit**. It analyzes resumes and job descriptions to provide a similarity score and offers suggestions to align resumes better with job requirements.

## 🚀 Features

- ✅ Extracts text from resumes and job descriptions (PDF/text)
- ✅ Identifies key phrases and skills using `spaCy` Matcher
- ✅ Generates semantic embeddings using Sentence Transformers
- ✅ Calculates similarity score between resume and job description
- ✅ Highlights matched and missing keywords
- ✅ Provides optimization suggestions
- ✅ User-friendly Streamlit interface

## 📁 Project Structure
- ├── main.py # Streamlit app interface and workflow
- ├── create_Embeddings.py # Embedding generation using transformer model
- ├── part1_functions.py # Text, chunk, and skill extraction functions
- ├── part2_functions.py # Similarity scoring and optimization logic


## ⚙️ How It Works

### 1. Text Extraction  
Extracts text from uploaded resumes and job descriptions using PyMuPDF or similar libraries.

### 2. Skill and Keyword Extraction  
Uses `spaCy` patterns to extract meaningful noun and adjective-based phrases.

### 3. Embedding Generation  
Converts extracted text into dense vectors using a pre-trained `SentenceTransformer` model.

### 4. Similarity Scoring  
Computes cosine similarity between resume and job description embeddings.

### 5. Suggestions  
Displays missing keywords/skills and offers suggestions to improve the resume for a higher match.

## 🛠 Requirements

- Python 3.8+
- streamlit  
- spacy  
- sentence-transformers  
- numpy  
- PyMuPDF  
- scikit-learn  
