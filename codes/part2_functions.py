import joblib
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy
import fitz

def load_components():
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    le = joblib.load("models/label_encoder.joblib")
    category_vectors = joblib.load("models/category_vectors.joblib")
    return tfidf, le, category_vectors

# # Using manual cosine similarity calculation
def cosine_similarity_manual(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    na = sum(a**2 for a in vec1) ** 0.5
    nb = sum(b**2 for b in vec2) ** 0.5

    if na == 0 or nb == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (na * nb)

def compute_similarity_score(resume_text, category, tfidf, le, category_vectors):
    cleaned_text = clean_resume(resume_text)
    user_vector = tfidf.transform([cleaned_text])
    cat_index = le.transform([category])[0]
    cat_vectors = category_vectors[cat_index]

    # Using inbuilt cosine_similarity function
    sim = cosine_similarity(user_vector, cat_vectors)

    # temp = cosine_similarity_manual(user_vector.toarray()[0], cat_vectors[0].toarray()[0])
    return sim.mean() * 100

nlp = spacy.load("en_core_web_sm")

def clean_resume(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", ' ', text)
    text = re.sub(r'\@\w+|\#', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_top_keywords(text, kw_model, top_n=10):
    return [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n, stop_words='english')]

def generate_suggestions(kw_model, resume_text, category):
    category_skill_map = {
        "Advocate": {
            "legal research", "litigation", "contract law", "legal writing", "case analysis", "intellectual property",
            "civil law", "criminal law", "corporate law", "negotiation"
        },
        "Arts": {
            "creative writing", "illustration", "graphic design", "adobe photoshop", "adobe illustrator",
            "drawing", "painting", "storytelling", "animation", "art history"
        },
        "Automation Testing": {
            "selenium", "cypress", "pytest", "automation framework", "testng", "java", "python",
            "jenkins", "jira", "ci/cd", "unit testing"
        },
        "Blockchain": {
            "solidity", "ethereum", "smart contracts", "web3.js", "blockchain", "cryptography",
            "decentralized apps", "truffle", "ganache", "nfts"
        },
        "Business Analyst": {
            "requirements gathering", "data analysis", "excel", "sql", "power bi", "tableau", "communication",
            "stakeholder management", "business process modeling", "agile"
        },
        "Civil Engineer": {
            "autocad", "staad pro", "construction management", "surveying", "project planning", "structural analysis",
            "site supervision", "estimation", "quality control", "ms project"
        },
        "Data Science": {
            "python", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "machine learning",
            "deep learning", "sql", "data visualization", "matplotlib", "seaborn", "nlp"
        },
        "Database": {
            "sql", "mysql", "oracle", "mongodb", "postgresql", "database design", "pl/sql",
            "normalization", "performance tuning", "backup and recovery"
        },
        "DevOps Engineer": {
            "docker", "kubernetes", "jenkins", "terraform", "aws", "azure", "linux", "ansible",
            "ci/cd", "bash scripting", "monitoring", "prometheus", "grafana"
        },
        "DotNet Developer": {
            "c#", ".net", "asp.net", "mvc", "sql server", "visual studio", "entity framework",
            "web api", "linq", "azure"
        },
        "ETL Developer": {
            "etl", "informatica", "talend", "data warehousing", "sql", "pl/sql", "ssrs", "ssis",
            "data modeling", "data pipelines"
        },
        "Electrical Engineering": {
            "matlab", "simulink", "pcb design", "microcontrollers", "circuit design", "embedded systems",
            "power systems", "autocad electrical", "labview", "plc"
        },
        "HR": {
            "recruitment", "hrms", "employee engagement", "performance management", "payroll", "talent acquisition",
            "training and development", "labor laws", "conflict resolution", "onboarding"
        },
        "Hadoop": {
            "hadoop", "hive", "pig", "mapreduce", "spark", "sqoop", "oozie", "big data", "hdfs", "yarn"
        },
        "Health and Fitness": {
            "personal training", "nutrition", "diet planning", "workout planning", "anatomy", "yoga",
            "cardio", "strength training", "health coaching", "first aid"
        },
        "Java Developer": {
            "java", "spring", "hibernate", "spring boot", "maven", "rest api", "microservices",
            "junit", "sql", "servlets"
        },
        "Mechanical Engineer": {
            "autocad", "solidworks", "catia", "ansys", "thermal engineering", "manufacturing", "mechanical design",
            "fluid mechanics", "matlab", "gd&t"
        },
        "Network Security Engineer": {
            "network security", "firewalls", "vpn", "ids/ips", "wireshark", "nmap", "linux", "ethical hacking",
            "penetration testing", "cisco"
        },
        "Operations Manager": {
            "operations management", "supply chain", "inventory management", "erp", "forecasting", "kpi tracking",
            "lean management", "six sigma", "vendor management", "logistics"
        },
        "PMO": {
            "project management", "pmo", "ms project", "jira", "scrum", "agile", "risk management",
            "project planning", "budgeting", "stakeholder communication"
        },
        "Python Developer": {
            "python", "flask", "django", "pandas", "numpy", "sqlalchemy", "rest api", "oop", "unit testing", "git"
        },
        "SAP Developer": {
            "sap abap", "sap fico", "sap hana", "sap mm", "sap sd", "bapi", "sap workflow", "idoc",
            "sap basis", "sap security"
        },
        "Sales": {
            "sales", "lead generation", "crm", "cold calling", "negotiation", "presentation", "target achievement",
            "b2b sales", "salesforce", "customer relationship"
        },
        "Testing": {
            "manual testing", "automation testing", "test cases", "bug tracking", "selenium", "jira",
            "testng", "black box testing", "white box testing", "sdlc"
        },
        "Web Designing": {
            "html", "css", "javascript", "bootstrap", "figma", "adobe xd", "responsive design",
            "ui/ux", "photoshop", "wireframing"
        }
    }
    resume_text = resume_text.lower()
    valid_skills = category_skill_map.get(category.title())
    keywords = []
    keywords = extract_top_keywords(resume_text, kw_model, top_n=20)
    suggestions = []
    for word in valid_skills:
        if word not in keywords and word not in suggestions:
            suggestions.append(word)
    return suggestions
