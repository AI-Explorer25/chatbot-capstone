import os
import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.optimizers import RMSprop

# ===============================
# Paths
# ===============================
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "..", "data")
models_dir = os.path.join(root_dir, "..", "models")
os.makedirs(models_dir, exist_ok=True)

# ===============================
# Load preprocessed data
# ===============================
encoder_input_data = np.load(os.path.join(data_dir, "encoder_input_data.npy"))
decoder_input_data = np.load(os.path.join(data_dir, "decoder_input_data.npy"))
decoder_target_data = np.load(os.path.join(data_dir, "decoder_target_data.npy"))

with open(os.path.join(data_dir, "input_tokenizer.pkl"), "rb") as f:
    input_tokenizer = pickle.load(f)
with open(os.path.join(data_dir, "target_tokenizer.pkl"), "rb") as f:
    target_tokenizer = pickle.load(f)

num_encoder_tokens = min(20000, len(input_tokenizer.word_index) + 1)
num_decoder_tokens = min(20000, len(target_tokenizer.word_index) + 1)

# ===============================
# Hyperparameters (reduced for faster training)
# ===============================
latent_dim = 64       # smaller for faster testing
embedding_dim = 64    # smaller embedding
batch_size = 32
epochs = 5            # fewer epochs for testing; increase later if needed

# ===============================
# Encoder
# ===============================
encoder_inputs = Input(shape=(None,), name="encoder_input")
enc_emb = Embedding(num_encoder_tokens, embedding_dim, name="encoder_embedding")(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# ===============================
# Decoder
# ===============================
decoder_inputs = Input(shape=(None,), name="decoder_input")
dec_emb = Embedding(num_decoder_tokens, embedding_dim, name="decoder_embedding")(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# ===============================
# Full model (training)
# ===============================
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=RMSprop(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# reshape targets for sparse_categorical_crossentropy
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# ===============================
# Train
# ===============================
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# ===============================
# Save full model
# ===============================
model.save(os.path.join(models_dir, "chatbot_model.keras"))
print("✅ Training complete and full model saved")

# ===============================
# Encoder inference model
# ===============================
encoder_model = Model(encoder_inputs, encoder_states, name="encoder_model")
encoder_model.save(os.path.join(models_dir, "encoder_model.keras"))
print("✅ Encoder model saved")

# ===============================
# Decoder inference model
# ===============================
decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_h")
decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_c")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb  # reuse embedding
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb, initial_state=decoder_states_inputs
)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_states2 = [state_h2, state_c2]

# Correct inference decoder
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2,
    name="decoder_model"
)
decoder_model.save(os.path.join(models_dir, "decoder_model.keras"))
print("✅ Decoder model saved")