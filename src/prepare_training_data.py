import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# ===============================
# Paths
# ===============================
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "..", "data")
combined_file = os.path.join(data_dir, "combined_chat_data.txt")

# ===============================
# CONFIG (CRITICAL FIXES)
# ===============================
MAX_SAMPLES = 50000          
MAX_VOCAB_SIZE = 20000       
MAX_LEN = 20                 

# ===============================
# Load data
# ===============================
input_texts = []
target_texts = []

with open(combined_file, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= MAX_SAMPLES:
            break

        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        input_text, target_text = parts

        # filter long sentences (CRITICAL)
        if len(input_text.split()) > MAX_LEN or len(target_text.split()) > MAX_LEN:
            continue

        target_text = "<START> " + target_text + " <END>"

        input_texts.append(input_text)
        target_texts.append(target_text)

print(f"Loaded {len(input_texts)} samples.")

# ===============================
# Tokenization (LIMITED VOCAB)
# ===============================
input_tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

target_tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)

# ===============================
# Padding
# ===============================
encoder_input_data = pad_sequences(input_sequences, maxlen=MAX_LEN, padding="post")
decoder_input_data = pad_sequences(target_sequences, maxlen=MAX_LEN, padding="post")

decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

# ===============================
# Save
# ===============================
np.save(os.path.join(data_dir, "encoder_input_data.npy"), encoder_input_data)
np.save(os.path.join(data_dir, "decoder_input_data.npy"), decoder_input_data)
np.save(os.path.join(data_dir, "decoder_target_data.npy"), decoder_target_data)

with open(os.path.join(data_dir, "input_tokenizer.pkl"), "wb") as f:
    pickle.dump(input_tokenizer, f)

with open(os.path.join(data_dir, "target_tokenizer.pkl"), "wb") as f:
    pickle.dump(target_tokenizer, f)

print("✅ Preprocessing complete")
print("Encoder shape:", encoder_input_data.shape)
print("Decoder shape:", decoder_input_data.shape)