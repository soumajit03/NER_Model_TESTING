import os
import re
import json
import streamlit as st
import torch
import pandas as pd
from transformers import BertForTokenClassification, BertTokenizerFast

# ========== CONFIG ========== #
MODEL_DIR = "./NERmodel-3"
KANDA_TEXT_DIR = "./TEXT_DATASETS/04_Kishkindha-kanda"
OUTPUT_JSON_DIR = "./corrected_json"
RETRAIN_JSON_DIR = "./retrain_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
FILES_PER_PAGE = 5
PUNCTUATIONS = {",", ".", "-", "–", "—", ";", ":", "!", "?", "'", "`", '"', "’", "‘"}

# ========== LOAD MODEL ========== #
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()
ID2LABEL = model.config.id2label
LABEL2ID = model.config.label2id
LABEL_OPTIONS = list(ID2LABEL.values())

# ========== HELPERS ========== #
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text.strip()) if s]

def tokenize_preserve_punctuation(text):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def predict_sentence_entities(sentence):
    tokens = tokenize_preserve_punctuation(sentence)
    results = []

    chunk_size = MAX_LEN - 2
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        encoding = tokenizer(
            chunk,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        encoding = {k: v.to(DEVICE) for k, v in encoding.items() if k != "offset_mapping"}

        with torch.no_grad():
            logits = model(**encoding).logits
            predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()

        prev_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != prev_word_idx:
                word = chunk[word_idx]
                label = ID2LABEL[predictions[idx]]

                # Fix: force punctuation or single 's' to O
                if word in PUNCTUATIONS or word.lower() == "s":
                    label = "O"

                results.append((word, label))
                prev_word_idx = word_idx

    return results

def convert_to_json_format(sentence_tokens, tags):
    entities = []
    char_index = 0
    for idx, (token, tag) in enumerate(zip(sentence_tokens, tags)):
        start = char_index
        end = start + len(token)
        entities.append([start, end, token, idx * 2, tag])
        char_index = end + 1  # for space
    sentence_text = " ".join(sentence_tokens) + "\r"
    return [sentence_text, {"entities": entities}]

# ========== STREAMLIT UI ========== #
st.set_page_config(page_title="Mythological NER Correction", layout="wide")
st.title("📜 Mythological NER - Tag Correction & Sentence-wise JSON")

st.subheader("🧠 Model Loaded From:")
st.code(MODEL_DIR)

st.subheader("📂 Kanda Text Files From:")
st.code(KANDA_TEXT_DIR)

text_files = sorted([f for f in os.listdir(KANDA_TEXT_DIR) if f.endswith(".txt")])
st.session_state.setdefault("file_index", 0)

start_idx = st.session_state["file_index"]
end_idx = min(start_idx + FILES_PER_PAGE, len(text_files))
current_files = text_files[start_idx:end_idx]

all_corrected = {}

if not current_files:
    st.warning("⚠️ No `.txt` files found.")
else:
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    for file in current_files:
        st.markdown(f"---\n### 📝 File: `{file}`")

        with open(os.path.join(KANDA_TEXT_DIR, file), "r", encoding="utf-8") as f:
            raw_text = f.read().strip()

        sentences = split_sentences(raw_text)
        full_json = []
        all_df_rows = []

        for sid, sentence in enumerate(sentences):
            pred = predict_sentence_entities(sentence)
            all_df_rows.extend(pred)
            full_json.append(convert_to_json_format(*zip(*pred)))

        df = pd.DataFrame(all_df_rows, columns=["word", "tag"])

        corrected_df = st.data_editor(
            df,
            column_config={
                "tag": st.column_config.SelectboxColumn("Tag", options=LABEL_OPTIONS, required=True)
            },
            num_rows="dynamic",
            key=f"editor_{file}"
        )

        all_corrected[file] = (corrected_df, sentences)

        if st.button(f"💾 Download JSON for `{file}`"):
            tokens = list(corrected_df["word"])
            tags = list(corrected_df["tag"])
            corrected_json = []

            cursor = 0
            for sentence in sentences:
                sentence_tokens = tokenize_preserve_punctuation(sentence)
                sentence_tags = tags[cursor: cursor + len(sentence_tokens)]
                corrected_json.append(convert_to_json_format(sentence_tokens, sentence_tags))
                cursor += len(sentence_tokens)

            json_path = os.path.join(OUTPUT_JSON_DIR, file.replace(".txt", ".json"))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(corrected_json, f, indent=2)

            st.success(f"✅ JSON saved to `{json_path}`")

# ========== Load More Button ========== #
if end_idx < len(text_files):
    if st.button("📂 Load Next 5 Files"):
        st.session_state["file_index"] += FILES_PER_PAGE
        st.experimental_rerun()

# ========== Retrain Button ========== #
if current_files and st.button("🚀 Retrain Model with All Corrected Tags"):
    st.info("Saving corrected JSONs for retraining...")
    os.makedirs(RETRAIN_JSON_DIR, exist_ok=True)

    for file, (df, original_sentences) in all_corrected.items():
        tokens = list(df["word"])
        tags = list(df["tag"])
        corrected_json = []

        cursor = 0
        for sentence in original_sentences:
            sentence_tokens = tokenize_preserve_punctuation(sentence)
            sentence_tags = tags[cursor: cursor + len(sentence_tokens)]
            corrected_json.append(convert_to_json_format(sentence_tokens, sentence_tags))
            cursor += len(sentence_tokens)

        retrain_path = os.path.join(RETRAIN_JSON_DIR, file.replace(".txt", ".json"))
        with open(retrain_path, "w", encoding="utf-8") as f:
            json.dump(corrected_json, f, indent=2)

    st.success(f"🎯 All sentence-wise JSONs saved in `{RETRAIN_JSON_DIR}`")
