# app.py
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from PIL import Image
import time

# ---------------- Page config ----------------
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="assets/favicon.png", layout="wide")

# ---------------- Ensure NLTK data is available (cached) ----------------
@st.cache_resource
def setup_nltk():
    """
    Ensure punkt, punkt_tab, and stopwords are available.
    If downloads fail (e.g., no internet), mark failure and caller can fallback.
    """
    success = {"punkt": False, "punkt_tab": False, "stopwords": False}
    try:
        nltk.data.find("tokenizers/punkt")
        success["punkt"] = True
    except LookupError:
        try:
            nltk.download("punkt")
            nltk.data.find("tokenizers/punkt")
            success["punkt"] = True
        except Exception:
            success["punkt"] = False

    try:
        nltk.data.find("tokenizers/punkt_tab/english")
        success["punkt_tab"] = True
    except LookupError:
        try:
            nltk.download("punkt_tab")
            nltk.data.find("tokenizers/punkt_tab/english")
            success["punkt_tab"] = True
        except Exception:
            success["punkt_tab"] = False

    try:
        nltk.data.find("corpora/stopwords")
        success["stopwords"] = True
    except LookupError:
        try:
            nltk.download("stopwords")
            nltk.data.find("corpora/stopwords")
            success["stopwords"] = True
        except Exception:
            success["stopwords"] = False

    return success

nltk_status = setup_nltk()

# ---------------- Helpers & Model load (cached) ----------------
@st.cache_resource
def load_vectorizer(path="vectorizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

ps = PorterStemmer()

# optional logo
def try_load_image(path):
    try:
        return Image.open(path)
    except Exception:
        return None

logo = try_load_image("assets/logo.png")

# ---------------- Safe tokenizer ----------------
def safe_tokenize(text: str):
    """
    Use nltk.word_tokenize if punkt/punkt_tab available.
    Otherwise fallback to a simple regex split or str.split.
    """
    if nltk_status.get("punkt") and nltk_status.get("punkt_tab"):
        try:
            return nltk.word_tokenize(text)
        except Exception:
            pass
    # fallback: simple split on whitespace & punctuation removal
    tokens = []
    for token in text.split():
        # remove leading/trailing punctuation
        tok = token.strip(string.punctuation)
        if tok:
            tokens.append(tok)
    return tokens

# ---------------- Preprocessing ----------------
def transform_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = safe_tokenize(text)

    filtered = []
    for token in tokens:
        if token.isalnum():
            # use stopwords only if available; otherwise assume a smaller stop set or skip removal
            if nltk_status.get("stopwords"):
                if token not in stopwords.words("english"):
                    filtered.append(ps.stem(token))
            else:
                # no stopwords resource — just stem alphanumeric tokens
                filtered.append(ps.stem(token))
    return " ".join(filtered)

# ---------------- UI ----------------
with st.container():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("📧 Email / SMS Spam Classifier")
        st.markdown("Enter message below and click **Predict**. If NLTK resources failed to download, a fallback tokenizer will be used.")
    with c2:
        if logo:
            st.image(logo, width=100)
        else:
            st.caption("Add logo.png to the folder to show a logo.")

# show NLTK status for debugging (optional)
st.info(f"NLTK status: {nltk_status}")

st.write("---")
left, right = st.columns([2, 3])

with left:
    input_sms = st.text_area("Enter message here:", height=220, placeholder="e.g. Congratulations — you won a prize! Click to claim...")
    if st.button("Predict"):
        if not input_sms.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Loading model & predicting..."):
                tfidf = load_vectorizer("vectorizer.pkl")
                model = load_model("model.pkl")

                pre = transform_text(input_sms)
                vector_input = tfidf.transform([pre])
                result = model.predict(vector_input)[0]

                # Try probabilities
                prob_spam = None
                try:
                    probs = model.predict_proba(vector_input)[0]
                    if hasattr(model, "classes_"):
                        classes = list(model.classes_)
                        if 1 in classes:
                            idx_spam = classes.index(1)
                            prob_spam = probs[idx_spam]
                        else:
                            prob_spam = probs[-1]
                    else:
                        prob_spam = probs[-1]
                except Exception:
                    prob_spam = None

                time.sleep(0.3)

                if result == 1:
                    st.markdown("### 🚨 Spam")
                    if prob_spam is not None:
                        st.write(f"Confidence Spam: {prob_spam:.2f}")
                    st.warning("Do not click links or download attachments.")
                else:
                    st.markdown("### ✅ Not Spam")
                    if prob_spam is not None:
                        st.write(f"Confidence Spam: {prob_spam:.2f}")

with right:
    st.subheader("Preprocessing preview")
    if input_sms.strip():
        st.write("**Original**")
        st.write(input_sms)
        st.write("**After preprocessing**")
        st.write(transform_text(input_sms))
    else:
        st.info("Enter a message to see preprocessing preview here.")

st.write("---")
st.caption("Run with: streamlit run app.py")