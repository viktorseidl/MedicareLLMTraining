import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle  # to load tokenizer if saved with pickle
from sklearn.metrics import classification_report, accuracy_score
# Load your trained model
model = tf.keras.models.load_model('best_model.h5')

# Load your tokenizer (adjust path and filename as needed)
with open('tokenizerevo.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max sequence length used during training
MAX_SEQUENCE_LENGTH = 10  # Adjust to your trained model's input length

# Your label dictionary (as provided)
labels = {
    0: "update_blutzucker",
    1: "update_blutdruck",
    2: "update_temperatur",
    3: "update_gewicht",
    4: "update_medikation",
    5: "update_ein_ausfuhr",
    6: "update_stuhlgang",
    7: "update_einzelbetreuung",
    8: "update_verbandswechsel ",
    9: "update_medikamenten_stellung",
    10: "update_nahrungsaufnahme",
    11: "update_fluessigkeitsaufnahme",
    12: "update_fixierung_bett",
    13: "update_remove_fixierung_bauchgurt",
    14: "update_gruppenbetreuung",
    15: "update_tagesgruppe",
    16: "update_medikamente_vorbereiten",
    17: "update_fixierung_rollstuhl",
    18: "update_remove_fixierung_rollstuhl",
    19: "update_fixierung_therapietisch",
    20: "update_remove_fixierung_therapietisch",
    21: "update_betreuung_43b",
    22: "update_behandlungspflege",
    23: "update_medikamente_bereitstellen",
    24: "update_beratende_gespraeche",
    25: "update_kg_extern",
    26: "update_mobilitaetsfaktor",
    27: "update_validierende_gespraeche",
    28: "update_lagerungsprotokoll",
    29: "update_duschen",
    30: "update_frisoer",
    31: "update_fusspflege",
    32: "update_nagelpflege",
    33: "update_haarwaesche",
    34: "update_fixierung_rollstuhlbremse_anlegen",
    35: "update_remove_fixierung_rollstuhlbremse",
    36: "update_toilettengang",
    37: "update_medikamentenkontrolle",
    38: "update_betreuung_beobachtung",
    39: "update_fixierung",
    40: "update_gewichtsauswertung",
    41: "update_medikamenten_vorbereitung_diabetiker",
    42: "update_medikamente_verabreichen_pfk"
}

# Test sentences (example)
test_sentences = [
    "Bitte aktualisiere den Blutzuckerwert auf 125 mg/dl bei Frau Müller.",
    "Trage die Temperatur von Herrn Schmidt mit 37,8 Grad ein.",
    "Aktualisiere das Gewicht von Frau Lehmann auf 68 Kilogramm."
]

# Preprocess input sentences
sequences = tokenizer.texts_to_sequences(test_sentences)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
max_vocab_index = 169  # Adjust to your embedding input_dim - 1
print("Max tokenizer index in input:", np.max(padded_sequences))
if np.max(padded_sequences) > max_vocab_index:
    print("Warning: input token index exceeds embedding vocab size. Clipping indices.")
    padded_sequences = np.where(padded_sequences > max_vocab_index, 0, padded_sequences)
# Predict with the model
predictions = model.predict(padded_sequences)

# Get predicted class indices
predicted_indices = np.argmax(predictions, axis=1)

# Map to label names
predicted_labels = [labels[idx] for idx in predicted_indices]

# Print results
for sentence, label in zip(test_sentences, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted label: {label}")
    print("---")
true_labels = [
    "update_blutzucker",
    "update_temperatur",
    "update_gewicht"
]
for sentence, pred_label in zip(test_sentences, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted label: {pred_label}")
    print() 
    print("Accuracy:", accuracy_score(true_labels, predicted_labels))
    print(classification_report(true_labels, predicted_labels, target_names=list(true_labels.values())))