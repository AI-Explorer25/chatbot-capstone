import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# Paths
# ===============================
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "..", "data")
models_dir = os.path.join(root_dir, "..", "models")

# ===============================
# Load inference models + tokenizers
# ===============================
encoder_model = load_model(os.path.join(models_dir, "encoder_model.keras"))
decoder_model = load_model(os.path.join(models_dir, "decoder_model.keras"))

with open(os.path.join(data_dir, "input_tokenizer.pkl"), "rb") as f:
    input_tokenizer = pickle.load(f)

with open(os.path.join(data_dir, "target_tokenizer.pkl"), "rb") as f:
    target_tokenizer = pickle.load(f)

# Reverse lookup for target words
reverse_target_index = {i: word for word, i in target_tokenizer.word_index.items()}
reverse_target_index[0] = ""

# Max lengths (inferred from training data)
encoder_maxlen = np.load(os.path.join(data_dir, "encoder_input_data.npy")).shape[1]
decoder_maxlen = np.load(os.path.join(data_dir, "decoder_input_data.npy")).shape[1]

# Special tokens
start_token = target_tokenizer.word_index.get("<start>", 1)
end_token = target_tokenizer.word_index.get("<end>", 2)

# ===============================
# Convert input text → padded sequence
# ===============================
def string_to_seq(text):
    seq = input_tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=encoder_maxlen, padding='post')
    return seq

# ===============================
# Sampling function: temperature + top-k
# ===============================
def sample_with_temperature_topk(preds, temperature=0.7, top_k=10):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    top_indices = np.argsort(preds)[-top_k:]
    top_probs = preds[top_indices]
    top_probs = top_probs / np.sum(top_probs)

    return np.random.choice(top_indices, p=top_probs)

# ===============================
# Step-by-step decoding using inference models
# ===============================
def generate_response(input_text, temperature=0.7, top_k=10, max_dec_steps=25):
    # Encode the input
    states_value = encoder_model.predict(string_to_seq(input_text), verbose=0)

    # Initialize target sequence with <start>
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    decoded_sentence = []

    for _ in range(max_dec_steps):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = sample_with_temperature_topk(output_tokens[0, -1, :], temperature, top_k)
        sampled_word = reverse_target_index.get(sampled_token_index, "")

        if sampled_token_index == end_token or sampled_word == "":
            break

        # Avoid including <start> or empty tokens
        if sampled_word != "<start>":
            decoded_sentence.append(sampled_word)

        # Update target sequence (next input token)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return " ".join(decoded_sentence)

# ===============================
# Chat loop
# ===============================
print("Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("> ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = generate_response(user_input, temperature=0.8, top_k=15)
    print(response)