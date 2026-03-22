import streamlit as st
# --- Page Setup ---
st.set_page_config(page_title="Smart Resume Matcher", page_icon="📄", layout="wide")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# Load Semantic Model (Cached so it only loads once)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()



st.markdown("""
# Smart Resume Matcher
This tool uses **Semantic AI (SBERT)** and **WordNet Synonyms** to understand the meaning behind your resume, not just the keywords.
""")

with st.sidebar:
    st.header("About")
    st.info("""
    - **Semantic Analysis:** Understands that "Led" and "Managed" mean the same thing.
    - **Keyword Gap:** Identifies missing skills even if you used different terminology.
    - **Noise Filtering:** Automatically ignores generic job-posting filler words.
    """)
    st.header("How It Works")
    st.write("1. Upload Resume (PDF)\n2. Paste Job Description\n3. Get AI-powered match score")

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Keep spaces to avoid merging words
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_synonyms(word):
    """Finds synonyms to ensure we don't penalize different wording."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    return synonyms

def extract_missing_keywords(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    processed_jd = job_description.replace("/", " ")
    job_words = set(clean_text(processed_jd).split())
    
    # 1. Expanded Noise Filter (Non-Skill Words)
    blacklist = {
        'company', 'candidate', 'role', 'overview', 'benefits', 'requirements', 
        'location', 'years', 'plus', 'pvt', 'ltd', 'ideal', 'looking', 'description',
        'employment', 'opportunity', 'skills', 'experience', 'responsibilities',
        'expertise', 'exposure', 'familiarity', 'hands', 'discription', 'bangalore', 
        'hybrid', 'india', 'candidate', 'qualified', 'success', 'working', 'ability','delivery','identify','combine'
    }
    
    # 2. Extract Nouns and Adjectives
    tagged_job = pos_tag(list(job_words))
    
    # We only want meaningful words that aren't in our blacklist
    target_keywords = [
        w for w, pos in tagged_job 
        if (pos.startswith('NN') or pos.startswith('JJ')) 
        and w not in blacklist 
        and len(w) > 3 # Ignore very short words like 'up' or 'in'
    ]
    
    missing_keywords = []
    for word in set(target_keywords):
        syns = get_synonyms(word)
        syns.add(word)
        
        # Check synonyms against the resume
        if not any(s in resume_words for s in syns):
            missing_keywords.append(word)
            
    return sorted(list(set(missing_keywords)))

def calculate_semantic_similarity(resume_text, job_description):
    """Calculates similarity based on the meaning of sentences."""
    embeddings = model.encode([resume_text, job_description])
    cosine_score = util.cos_sim(embeddings[0], embeddings[1])
    score = float(cosine_score[0][0]) * 100
    return round(score, 2)

# --- Main App ---

def main():
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
    job_description = st.text_area("Paste the job description", height=250)

    if st.button("Analyze Match"):
        if not uploaded_file or not job_description:
            st.warning("Please provide both a resume and a job description.")
            return
        
        with st.spinner("AI is analyzing semantic context and filtering noise..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if not resume_text:
                st.error("Could not extract text. Please try a different PDF.")
                return 
            
            # 1. Calculate Score
            similarity_score = calculate_semantic_similarity(resume_text, job_description)

            # 2. Extract Filtered Missing Keywords
            missing = extract_missing_keywords(resume_text, job_description)

            # --- Display Results ---
            st.subheader("Analysis Results")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Match Score", f"{similarity_score}%")
                fig, ax = plt.subplots(figsize=(6, 1))
                colors = ['#ff4b4b', '#ffa726', '#0f9d58']
                color_index = min(int(similarity_score // 33), 2)
                ax.barh([0], [similarity_score], color=colors[color_index])
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                st.pyplot(fig)

            with col2:
                if similarity_score < 40:
                    st.warning("⚠️ **Low Match:** Consider a major resume overhaul for this role.")
                elif similarity_score < 70:
                    st.info("💡 **Good Match:** Your profile is strong. Add the key concepts below to boost your score.")
                else:
                    st.success("✅ **Excellent Match:** Your resume is highly aligned with this position!")

            # --- Readable Optimization Tips ---
            if missing:
                st.divider()
                st.subheader("🎯 Optimization Tips")
                st.write("To increase your relevance, consider incorporating these missing concepts into your resume:")

                # Categorize: High-value keywords (longer/technical words) vs others
                high_value = [w for w in missing if len(w) > 5]
                other_terms = [w for w in missing if len(w) <= 5]

                # Display High-Value Keywords prominently
                st.markdown("### 🛠 Key Skills & Tools")
                # Format as bold, uppercase tags
                formatted_tags = "   •   ".join([f"**{w.upper()}**" for w in high_value[:15]])
                st.write(formatted_tags)

                # Use an expander for the full list to keep UI clean
                with st.expander("View all missing terminology"):
                    st.write(", ".join(missing))

if __name__ == "__main__":
    main()