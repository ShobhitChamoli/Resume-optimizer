import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import tempfile
import os
import docx2txt
import re
from pdfminer.high_level import extract_text as extract_pdf_text
from collections import defaultdict
import numpy as np
from sentence_transformers import util

def load_models():
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        from spacy.cli import download
        download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model)

    return nlp, model, kw_model

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()    
    return text

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = extract_pdf_text(tmp_path)
        os.remove(tmp_path)
        return text
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        text = docx2txt.process(tmp_path)
        os.remove(tmp_path)
        return text
    else:
        return uploaded_file.read().decode('utf-8', errors='ignore')

def extract_semantic_chunks(text, nlp, chunk_size=300):
    doc = nlp(text)
    chunks = []
    temp = ""
    for sent in doc.sents:
        temp += sent.text.strip() + " "
        if len(temp) >= chunk_size:
            chunks.append(temp.strip())
            temp = ""
    if temp:
        chunks.append(temp.strip())
    return chunks

def extract_skills_with_pos(doc, nlp):
    from spacy.matcher import Matcher
    matcher = Matcher(nlp.vocab)
    pattern = [{"POS": {"IN": ["NOUN", "PROPN", "ADJ"]}}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "?"}]
    matcher.add("SKILL_PHRASE", [pattern])
    matches = matcher(doc)
    phrases = set()
    for _, start, end in matches:
        span = doc[start:end]
        if len(span.text.strip()) > 3:
            phrases.add(span.text.lower())
    return list(phrases)

def analyze_job_description(jd_text, nlp, kw_model):
    jd_text = clean_text(jd_text)
    doc = nlp(jd_text)
    entities = defaultdict(list)

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "TECH", "SKILL", "DOMAIN"]:
            entities[ent.label_].append(ent.text)

    tech_keywords = [
        "python", "flask", "react", "aws", "docker", "kubernetes", "numpy", "pandas",
        "django", "fastapi", "node.js", "express", "typescript", "javascript",
        "html", "css", "mongodb", "postgresql", "mysql", "redis", "git",
        "github", "gitlab", "ci/cd", "jenkins", "tensorflow", "pytorch",
        "scikit-learn", "matplotlib", "seaborn", "linux", "bash", "azure",
        "gcp", "restapi", "graphql", "firebase", "lambda", "s3", "terraform",
        "ansible", "selenium", "beautifulsoup", "openai", "huggingface"
    ]
    for kw in tech_keywords:
        if kw.lower() in jd_text.lower():
            entities["SKILL"].append(kw)

    requirements = []
    for sent in doc.sents:
        if "require" in sent.text.lower() or "must have" in sent.text.lower():
            requirements.append(sent.text.strip())

    key_phrases = [kw[0] for kw in kw_model.extract_keywords(jd_text, top_n=15)]

    return {
        "entities": dict(entities),
        "requirements": requirements,
        "semantic_chunks": extract_semantic_chunks(jd_text, nlp),
        "keywords": key_phrases
    }

def analyze_resume(resume_text, nlp):
    resume_text = clean_text(resume_text)
    doc = nlp(resume_text)
    sections = {"experience": [], "skills": [], "education": [], "projects": []}
    lines = resume_text.split('\n')
    current_section = None
    for line in lines:
        line_lower = line.lower().strip()
        if "experience" in line_lower:
            current_section = "experience"
        elif "skills" in line_lower:
            current_section = "skills"
        elif "education" in line_lower:
            current_section = "education"
        elif "projects" in line_lower:
            current_section = "projects"
        if current_section and line.strip():
            sections[current_section].append(line.strip())

    skills = extract_skills_with_pos(doc, nlp)
    return {
        "sections": sections,
        "skills": list(set(skills)),
        "semantic_chunks": extract_semantic_chunks(resume_text, nlp)
    }

def semantic_match(jd_chunks, resume_chunks, resume_sections, model):
    jd_embeddings = model.encode(jd_chunks, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_chunks, convert_to_tensor=True)
    cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings)

    weights = []
    joined = "\n".join(resume_sections["experience"] + resume_sections["skills"])
    for chunk in resume_chunks:
        weights.append(1.2 if chunk in joined else 1.0)

    top_matches = []
    for i in range(len(jd_chunks)):
        scores = cosine_scores[i] * np.array(weights)
        top_idx = np.argmax(scores)
        top_matches.append({
            "jd_chunk": jd_chunks[i],
            "resume_chunk": resume_chunks[top_idx],
            "score": scores[top_idx].item()
        })

    overall_score = sum(match["score"] for match in top_matches) / len(top_matches)
    return {
        "overall_score": overall_score * 100,
        "matches": sorted(top_matches, key=lambda x: x["score"], reverse=True)
    }

def generate_suggestions(jd_analysis, resume_analysis, matches):
    suggestions = []
    valid_skills = set([
        "python", "flask", "django", "react", "angular", "node.js", "express", "mongodb", "sql", "mysql",
        "aws", "azure", "docker", "kubernetes", "tensorflow", "pytorch", "nlp", "machine learning",
        "deep learning", "html", "css", "javascript", "typescript", "pandas", "numpy", "git", "github",
        "linux", "bash", "java", "c++", "c#", "r", "matlab", "power bi", "tableau", "hadoop", "spark",
        "mern", "mevn", "rest api", "graphql", "firebase", "postman", "selenium", "jest", "mocha", "ci/cd"
    ])
    jd_entities = {e.lower() for ent in jd_analysis["entities"].values() for e in ent if e.lower() in valid_skills}
    resume_entities = set([s.lower() for s in resume_analysis["skills"]])
    missing_entities = jd_entities - resume_entities

    if missing_entities:
        suggestions.append({
            "title": "Missing Skills/Technologies",
            "description": "These are mentioned in the job description but not found in your resume.",
            "items": list(missing_entities)
        })

    covered_reqs = set()
    for m in matches:
        for req in jd_analysis["requirements"]:
            if req in m["jd_chunk"]:
                covered_reqs.add(req)
    missing_reqs = set(jd_analysis["requirements"]) - covered_reqs

    if missing_reqs:
        suggestions.append({
            "title": "Missing Explicit Requirements",
            "description": "These explicit JD requirements weren't addressed clearly.",
            "items": list(missing_reqs)
        })
    return suggestions
