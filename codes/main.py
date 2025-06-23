import streamlit as st
from part1_functions import (
    load_models, extract_text, analyze_job_description, analyze_resume,
    semantic_match, generate_suggestions as generate_suggestions1
)
from part2_functions import (
    load_components, compute_similarity_score, extract_text_from_pdf,
    generate_suggestions as generate_suggestions2
)

st.set_page_config(page_title="Unified Resume Optimiser", layout="wide")
st.title("Resume Optimiser")

option = st.radio("Choose analysis type:", ["Resume - Job Description Analysis", "Resume - Job Category Analysis"])
@st.cache_resource
def get_models():
    return load_models()

nlp, model, kw_model = get_models()

def run_ro_part2():
    import nltk

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

    st.title("📄 Resume Optimiser")

    tfidf, le, category_vectors = load_components()
    category_options = le.classes_
    selected_category = st.selectbox("Select the job category you are targeting:", category_options)
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

    if uploaded_file and selected_category:
        raw_text = extract_text_from_pdf(uploaded_file)
        relevance_score = compute_similarity_score(raw_text, selected_category, tfidf, le, category_vectors)

        st.subheader("Resume Optimiser")
        st.write(f"{relevance_score:.2f}% relevant to category: *{selected_category}*")

        suggestions = generate_suggestions2(kw_model, raw_text, selected_category)
        if suggestions:
            st.markdown("## Suggestions for Improvement")
            st.markdown(suggestions)


def run_ro_part1():

    st.header("Resume scorer and Optimiser (Suggestion based)")
    st.write("Understand your resume’s alignment with a job description using deep NLP techniques.")

    col1, col2 = st.columns(2)
    with col1:
        jd_file = st.file_uploader("Upload Job Description", type=['pdf', 'docx', 'txt'])
    with col2:
        resume_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])

    if st.button("Analyze", type="primary"):
        if not jd_file or not resume_file:
            st.error("Please upload both files.")
            return

        with st.spinner("Analyzing..."):
            jd_text = extract_text(jd_file)
            resume_text = extract_text(resume_file)

            jd_analysis = analyze_job_description(jd_text, nlp, kw_model)
            resume_analysis = analyze_resume(resume_text, nlp)

            match_results = semantic_match(
                jd_analysis["semantic_chunks"],
                resume_analysis["semantic_chunks"],
                resume_analysis["sections"],
                model
            )
            suggestions = generate_suggestions1(
                jd_analysis, resume_analysis, match_results["matches"]
            )

            st.subheader(f"Match Score: {match_results['overall_score']:.1f}%")
            st.progress(match_results['overall_score'] / 100)

            st.markdown("## Suggestions for Improvement")
            if suggestions:
                for s in suggestions:
                    st.markdown(f"### {s['title']}")
                    st.write(s['description'])
                    for item in s['items']:
                        st.write(f"- {item}")

if option == "Resume - Job Description Analysis":
    run_ro_part1()
elif option == "Resume - Job Category Analysis":
    run_ro_part2()
