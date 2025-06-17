# import streamlit as st
# import trafilatura
# import nltk
# from nltk.tokenize import word_tokenize
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
# import requests
# import json
# import math
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# from string import punctuation
# from collections import defaultdict
# import heapq

# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)

# # Set page config
# st.set_page_config(page_title="Smart Web Summarizer", layout="wide", page_icon="🧠")

# # ===================== Tailwind-Styled Title =====================
# st.markdown("""
#     <div class="text-center py-4">
#         <h1 class="text-4xl font-bold text-blue-700 animate-pulse">🧠 Smart Web Summarizer</h1>
#         <p class="text-lg text-gray-600">Extractive, Abstractive, Hybrid, and LLaMA-3 Powered Summaries</p>
#     </div>
# """, unsafe_allow_html=True)

# # ===================== Summarizer Class =====================
# class AdvancedSummarizer:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english')).union(set(punctuation))
#         self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#         self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def extractive_summarize(self, text, ratio=0.4):
#         sentences = sent_tokenize(text)
#         if len(sentences) <= 1:
#             return text

#         word_freq = defaultdict(int)
#         sentence_words = []

#         for sentence in sentences:
#             words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in self.stop_words and word.isalpha()]
#             sentence_words.append(words)
#             for word in words:
#                 word_freq[word] += 1

#         if not word_freq:
#             return sentences[0]

#         num_sentences = len(sentences)
#         idf = {word: 1 + math.log(num_sentences / (1 + sum(word in words for words in sentence_words))) for word in word_freq}

#         sent_scores = defaultdict(float)
#         for i, words in enumerate(sentence_words):
#             for word in words:
#                 tf = words.count(word) / len(words)
#                 sent_scores[i] += tf * idf[word]

#         top_indices = heapq.nlargest(max(1, int(num_sentences * ratio)), sent_scores, key=sent_scores.get)
#         best_sentences = [sentences[i] for i in sorted(top_indices)]
#         return ' '.join(best_sentences)

#     def abstractive_summarize(self, text, max_length=150, min_length=50):
#         if len(text) < 30:
#             return text

#         inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         summary_ids = self.model.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             min_length=min_length,
#             length_penalty=2.0,
#             num_beams=4,
#             early_stopping=True
#         )
#         return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     def hybrid_summarize(self, text, extract_ratio=0.5):
#         extractive = self.extractive_summarize(text, extract_ratio)
#         return self.abstractive_summarize(extractive)

# # ===================== GROQ Call =====================
# def summarize_with_groq(text):
#     try:
#         api_key = "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"  # Replace this
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": "llama3-8b-8192",
#             "messages": [
#                 {"role": "system", "content": "You are a helpful summarizer."},
#                 {"role": "user", "content": f"Summarize the following text in 150 words or fewer:\n\n{text[:3000]}"}
#             ],
#             "temperature": 0.5
#         }
#         response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(payload))
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         return f"[Groq Error] {e}"

# # ===================== Web Scraping =====================
# def fetch_main_text_trafilatura(url):
#     try:
#         html = trafilatura.fetch_url(url)
#         return trafilatura.extract(html)
#     except:
#         return None

# # ===================== UI =====================
# with st.form("url_form"):
#     url = st.text_input("Enter Article URL:", placeholder="https://example.com/article")
#     submitted = st.form_submit_button("Summarize")

# if submitted and url:
#     with st.spinner("Fetching and summarizing content..."):
#         content = fetch_main_text_trafilatura(url)
#         if not content:
#             st.error("Failed to fetch or extract article content.")
#         else:
#             summarizer = AdvancedSummarizer()
#             extractive = summarizer.extractive_summarize(content)
#             abstractive = summarizer.abstractive_summarize(content)
#             hybrid = summarizer.hybrid_summarize(content)
#             groq_summary = summarize_with_groq(content)

#             tab1, tab2, tab3, tab4 = st.tabs(["📝 Extractive", "🧠 Abstractive", "🔀 Hybrid", "🌐 Groq (LLM)"])

#             with tab1:
#                 st.subheader("Extractive Summary")
#                 st.success(extractive)
#             with tab2:
#                 st.subheader("Abstractive Summary")
#                 st.info(abstractive)
#             with tab3:
#                 st.subheader("Hybrid Summary")
#                 st.warning(hybrid)
#             with tab4:
#                 st.subheader("Groq LLaMA Summary")
#                 st.error(groq_summary)

#             st.balloons()
# else:
#     st.markdown("<p class='text-gray-500 text-center'>👆 Paste a valid URL to get started.</p>", unsafe_allow_html=True)

# ====================================second

# import streamlit as st
# import trafilatura
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from string import punctuation
# from collections import defaultdict
# import heapq
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import math
# import requests
# import json

# # Download required NLTK data
# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)

# # ==================== Page Config ====================
# st.set_page_config(page_title="Smart Web Summarizer", layout="wide", page_icon="🧠")

# # ==================== Tailwind-style Header ====================
# st.markdown("""
# <div style="text-align:center; padding: 20px;">
#     <h1 style="font-size:3em; color:#1e3a8a; animation: pulse 2s infinite;">🧠 Smart Web Summarizer</h1>
#     <p style="color: #6b7280; font-size: 1.2em;">Summarize any webpage using Extractive, Abstractive, Hybrid & LLaMA-3 powered Groq LLM</p>
# </div>
# """, unsafe_allow_html=True)

# # ==================== GROQ API Summarizer ====================
# def summarize_with_groq(text):
#     try:
#         api_key = "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"  # Replace this with your Groq key
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": "llama3-8b-8192",
#             "messages": [
#                 {"role": "system", "content": "You are a helpful summarizer."},
#                 {"role": "user", "content": f"Summarize the following text in 150 words or fewer:\n\n{text[:3000]}"}
#             ],
#             "temperature": 0.5
#         }
#         response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(payload))
#         response.raise_for_status()
#         return response.json()["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         return f"[Groq Error] {e}"

# # ==================== Extractor ====================
# def fetch_main_text(url: str) -> str | None:
#     try:
#         html = trafilatura.fetch_url(url)
#         return trafilatura.extract(html)
#     except Exception as e:
#         return None

# # ==================== Summarizer Class ====================
# class SmartSummarizer:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english')).union(set(punctuation))
#         self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#         self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def extractive_summarize(self, text, ratio=0.4):
#         sentences = sent_tokenize(text)
#         if len(sentences) <= 1:
#             return text

#         word_freq = defaultdict(int)
#         sentence_words = []

#         for sentence in sentences:
#             words = [word.lower() for word in word_tokenize(sentence)
#                      if word.lower() not in self.stop_words and word.isalpha()]
#             sentence_words.append(words)
#             for word in words:
#                 word_freq[word] += 1

#         num_sentences = len(sentences)
#         idf = {word: 1 + math.log(num_sentences / (1 + sum(word in words for words in sentence_words)))
#                for word in word_freq}

#         sent_scores = defaultdict(float)
#         for i, words in enumerate(sentence_words):
#             for word in words:
#                 tf = words.count(word) / len(words)
#                 sent_scores[i] += tf * idf[word]

#         top_indices = heapq.nlargest(max(1, int(num_sentences * ratio)), sent_scores, key=sent_scores.get)
#         selected = [sentences[i] for i in sorted(top_indices)]
#         return ' '.join(selected)

#     def abstractive_summarize(self, text, max_length=150, min_length=50):
#         if len(text) < 30:
#             return text
#         inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         summary_ids = self.model.generate(
#             inputs["input_ids"],
#             max_length=max_length,
#             min_length=min_length,
#             length_penalty=2.0,
#             num_beams=4,
#             early_stopping=True
#         )
#         return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     def hybrid_summarize(self, text, extract_ratio=0.5):
#         extractive = self.extractive_summarize(text, extract_ratio)
#         return self.abstractive_summarize(extractive)

# # ==================== Streamlit UI ====================
# with st.form("summarizer_form"):
#     url = st.text_input("🔗 Enter Article URL:", placeholder="https://example.com/article")
#     submit = st.form_submit_button("🔍 Summarize Now")

# if submit:
#     with st.spinner("🔄 Fetching and summarizing content..."):
#         content = fetch_main_text(url)
#         if not content:
#             st.error("❌ Could not extract content. Please check the URL.")
#         else:
#             summarizer = SmartSummarizer()
#             extractive = summarizer.extractive_summarize(content)
#             abstractive = summarizer.abstractive_summarize(content)
#             hybrid = summarizer.hybrid_summarize(content)
#             groq_summary = summarize_with_groq(content)

#             tab1, tab2, tab3, tab4 = st.tabs(["📝 Extractive", "🧠 Abstractive", "🔀 Hybrid", "🌐 Groq LLM"])

#             with tab1:
#                 st.subheader("📌 Extractive Summary")
#                 st.success(extractive)

#             with tab2:
#                 st.subheader("🧠 Abstractive Summary")
#                 st.info(abstractive)

#             with tab3:
#                 st.subheader("🤖 Hybrid Summary")
#                 st.warning(hybrid)

#             with tab4:
#                 st.subheader("🌐 Groq LLaMA-3 Summary")
#                 st.error(groq_summary)

#             st.balloons()
# else:
#     st.markdown("<p style='text-align:center; color:gray;'>👆 Enter a valid URL and click 'Summarize Now'</p>", unsafe_allow_html=True)



# ==================try new

import streamlit as st
import trafilatura
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import requests
import json
import math
from string import punctuation
from collections import defaultdict
import heapq

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Set page config
st.set_page_config(page_title="Smart Web Summarizer", layout="wide", page_icon="🧠")

# ===================== Title =====================
st.markdown("""
    <div style="text-align: center; padding: 20px 0">
        <h1 style="font-size: 40px; color: #2563eb; animation: pulse 2s infinite">🧠 Smart Web Summarizer</h1>
        <p style="color: #4b5563">Extractive, Abstractive, Hybrid, and LLaMA-3 Powered Summaries</p>
    </div>
""", unsafe_allow_html=True)

# ===================== Summarizer Class =====================
class AdvancedSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')).union(set(punctuation))
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extractive_summarize(self, text, ratio=0.4):
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text

        word_freq = defaultdict(int)
        sentence_words = []

        for sentence in sentences:
            words = [word.lower() for word in word_tokenize(sentence)
                     if word.lower() not in self.stop_words and word.isalpha()]
            sentence_words.append(words)
            for word in words:
                word_freq[word] += 1

        if not word_freq:
            return sentences[0]

        num_sentences = len(sentences)
        idf = {word: 1 + math.log(num_sentences / (1 + sum(word in words for words in sentence_words)))
               for word in word_freq}

        sent_scores = defaultdict(float)
        for i, words in enumerate(sentence_words):
            for word in words:
                tf = words.count(word) / len(words)
                sent_scores[i] += tf * idf[word]

        top_indices = heapq.nlargest(max(1, int(num_sentences * ratio)), sent_scores, key=sent_scores.get)
        best_sentences = [sentences[i] for i in sorted(top_indices)]
        return ' '.join(best_sentences)

    def abstractive_summarize(self, text, max_length=150, min_length=50):
        if len(text) < 30:
            return text

        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def hybrid_summarize(self, text, extract_ratio=0.5):
        extractive = self.extractive_summarize(text, extract_ratio)
        return self.abstractive_summarize(extractive)

# ===================== GROQ Call =====================
def summarize_with_groq(text):
    try:
        api_key = "gsk_YOUR_KEY_HERE"  # Replace with actual key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful summarizer."},
                {"role": "user", "content": f"Summarize the following text in 150 words or fewer:\n\n{text[:3000]}"}
            ],
            "temperature": 0.5
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Groq Error] {e}"

# ===================== Web Scraper =====================
def fetch_main_text_trafilatura(url):
    try:
        html = trafilatura.fetch_url(url)
        return trafilatura.extract(html)
    except:
        return None

# ===================== Evaluation =====================
def evaluate_summary(summary, original):
    word_count = len(word_tokenize(summary))
    original_len = len(word_tokenize(original))
    reduction_score = min(10, (1 - word_count / original_len) * 100 / 10)

    sentences = sent_tokenize(summary)
    readability_score = 8 + (1 if all(s[0].isupper() for s in sentences[:2]) else 0)
    relevance_score = 9 if "http" not in summary else 6
    fluency_score = 9 if summary[-1] in ".!?" else 6
    coverage_score = 7 + (1 if len(sentences) > 2 else 0) + (1 if word_count > 40 else 0)

    total = reduction_score + coverage_score + readability_score + relevance_score + fluency_score

    return {
        "Words": word_count,
        "Conciseness": round(reduction_score, 2),
        "Coverage": coverage_score,
        "Readability": readability_score,
        "Relevance": relevance_score,
        "Fluency": fluency_score,
        "Total Score": round(total, 2)
    }

# ===================== Streamlit UI =====================
with st.form("url_form"):
    url = st.text_input("Enter Article URL:", placeholder="https://example.com/article")
    submitted = st.form_submit_button("Summarize")

if submitted and url:
    with st.spinner("Fetching and summarizing content..."):
        content = fetch_main_text_trafilatura(url)
        if not content:
            st.error("Failed to fetch or extract article content.")
        else:
            summarizer = AdvancedSummarizer()
            extractive = summarizer.extractive_summarize(content)
            abstractive = summarizer.abstractive_summarize(content)
            hybrid = summarizer.hybrid_summarize(content)
            groq_summary = summarize_with_groq(content)

            tab1, tab2, tab3, tab4 = st.tabs(["📝 Extractive", "🧠 Abstractive", "🔀 Hybrid", "🌐 Groq (LLM)"])
            with tab1:
                st.subheader("Extractive Summary")
                st.success(extractive)
            with tab2:
                st.subheader("Abstractive Summary")
                st.info(abstractive)
            with tab3:
                st.subheader("Hybrid Summary")
                st.warning(hybrid)
            with tab4:
                st.subheader("Groq LLaMA Summary")
                st.error(groq_summary)

            st.markdown("---")
            st.subheader("📊 Summary Score Evaluation")

            results = {
                "Extractive": evaluate_summary(extractive, content),
                "Abstractive": evaluate_summary(abstractive, content),
                "Hybrid": evaluate_summary(hybrid, content),
                "Groq": evaluate_summary(groq_summary, content)
            }

            score_df = []
            for name, metrics in results.items():
                row = {"Method": name}
                row.update(metrics)
                score_df.append(row)

            import pandas as pd
            score_df = pd.DataFrame(score_df).sort_values(by="Total Score", ascending=False)
            st.dataframe(score_df, use_container_width=True)

            st.success(f"🏆 Best Method: {score_df.iloc[0]['Method']} with Score: {score_df.iloc[0]['Total Score']}/50")

else:
    st.markdown("<p style='text-align: center; color: #6b7280;'>👆 Paste a valid URL to get started.</p>", unsafe_allow_html=True)
