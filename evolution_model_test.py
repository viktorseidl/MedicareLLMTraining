import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score

# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Load the tokenizer
with open('tokenizerevo.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define max sequence length
MAX_SEQUENCE_LENGTH = 10

# Label dictionary
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

# Inverse label map for lookup
label_name_to_index = {v.strip(): k for k, v in labels.items()}  # strip accidental trailing spaces

# Test input
test_sentences = [
    "Bitte aktualisiere den Blutzuckerwert auf 125 mg/dl bei Frau Müller.",
  "Trage die Temperatur von Herrn Schmidt mit 37,8 Grad ein.",
  "Aktualisiere das Gewicht von Frau Lehmann auf 68 Kilogramm.",
  "Dokumentiere den Verbandswechsel bei Herrn Braun heute um 10 Uhr.",
  "Notiere, dass Frau Schulz ihre Medikamente um 14 Uhr erhalten hat.",
  "Pflegefachkraft hat die Medikamente bei Herrn Meier verabreicht – bitte eintragen.",
  "Füge einen Termin zur Einzelbetreuung für Frau König um 15 Uhr hinzu.",
  "Ergänze, dass die Gruppentherapie mit Bewohnern aus Wohnbereich 2 stattgefunden hat.",
  "Fixierung im Rollstuhl bei Herrn Neumann wurde angelegt – bitte dokumentieren.",
  "Bitte trage die Medikamentenvorbereitung für Diabetikerin Frau Schulze ein.",
  "Ein- und Ausfuhr bei Herrn Weber: Einfuhr 1200 ml, Ausfuhr 1100 ml.",
  "Eintrag: Fixierung am Therapietisch bei Frau Krüger durchgeführt.",
  "Update: Bewohnerin Frau Wagner wurde heute geduscht.",
  "Nagelpflege bei Herrn Schröder durchgeführt – bitte notieren.",
  "Fußpflege bei Frau Hartmann abgeschlossen – trage es ein.",
  "Temperatur bei Herrn Schwarz auf 38,2 aktualisieren.",
  "Verabreichung von Insulin bei Frau Becker dokumentieren.",
  "Fixierung mit Bauchgurt im Bett bei Herrn Kunze aktiv – bitte eintragen.",
  "Fixierung Bauchgurt im Rollstuhl wurde entfernt bei Frau Peters.",
  "Trage validierendes Gespräch mit Frau Nowak heute um 11:30 Uhr ein.",
  "Medikamente für Frau Jansen wurden heute gestellt – bitte eintragen.",
  "Die KG wurde extern bei Herrn Fischer heute erledigt – dokumentieren.",
  "Aktualisiere Mobilitätsfaktor für Frau Klein auf 1,7.",
  "Unsere Pflegefachkraft hat die Medikamentenkontrolle bei Herrn Walter durchgeführt.",
  "Füge eine neue Behandlungspflege bei Bewohnerin Schmidt hinzu.",
  "Beratendes Gespräch mit Angehörigen von Frau Brandt eintragen.",
  "Lagerung bei Herrn Dietrich wurde um 9 Uhr durchgeführt – bitte notieren.",
  "Eintrag: Nahrungsaufnahme bei Frau Lange normal.",
  "Einzelbetreuung bei Frau Keller von 13 bis 14 Uhr dokumentieren.",
  "Friseurtermin bei Herrn Ludwig heute abgeschlossen – bitte notieren.",
  "Neue Fixierung am Therapietisch bei Frau Vogt aktiviert.",
  "Bereite Medikamente für Herrn Scholz vor – eintragen.",
  "Aktualisiere Eintrag: Medikamente verabreicht bei Frau Bergmann um 16 Uhr.",
  "Einfuhr 1500 ml bei Herrn Otto – bitte dokumentieren.",
  "Update: Bauchgurt im Bett bei Frau Seidel wurde entfernt.",
  "Dokumentiere bitte, dass die Gruppenbetreuung heute stattgefunden hat.",
  "Medikamente bei Frau Keller vorbereitet – bitte im System speichern.",
  "Duschen bei Herrn Pfeiffer abgeschlossen – eintragen.",
  "Fixierung an Bauchgurt im Rollstuhl bei Frau Frank erneut angelegt.",
  "Trage ein: Medikamentengabe bei Herrn Paulsen durch PFK erfolgt."
] 
# True labels for above test samples
true_labels = [
     "update_blutdruck",
  "update_temperatur",
  "update_gewicht",
  "update_verbandswechsel",
  "update_medikamenten_stellung",
  "update_medikamente_verabreichen_pfk",
  "update_einzelbetreuung",
  "update_gruppenbetreuung",
  "update_fixierung_rollstuhl",
  "update_medikamenten_vorbereitung_diabetiker",
  "update_ein_ausfuhr",
  "update_fixierung_therapietisch",
  "update_duschen",
  "update_nagelpflege",
  "update_fusspflege",
  "update_temperatur",
  "update_medikamenten_vorbereitung_diabetiker",
  "update_fixierung_bett",
  "update_remove_fixierung_rollstuhl",
  "update_validierende_gespraeche",
  "update_medikamenten_stellung",
  "update_kg_extern",
  "update_mobilitaetsfaktor",
  "update_medikamentenkontrolle",
  "update_behandlungspflege",
  "update_beratende_gespraeche",
  "update_lagerungsprotokoll",
  "update_nahrungsaufnahme",
  "update_einzelbetreuung",
  "update_frisoer",
  "update_fixierung_therapietisch",
  "update_medikamente_vorbereiten",
  "update_medikamenten_stellung",
  "update_ein_ausfuhr",
  "update_remove_fixierung_bauchgurt",
  "update_gruppenbetreuung",
  "update_medikamente_vorbereiten",
  "update_duschen",
  "update_fixierung_rollstuhl",
  "update_medikamente_verabreichen_pfk"
]

# Encode input
sequences = tokenizer.texts_to_sequences(test_sentences)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Optional check for vocab size
max_vocab_index = 169
if np.max(padded_sequences) > max_vocab_index:
    print("Warning: input token index exceeds embedding vocab size. Clipping indices.")
    padded_sequences = np.where(padded_sequences > max_vocab_index, 0, padded_sequences)

# Predict
predictions = model.predict(padded_sequences)
predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = [labels[idx].strip() for idx in predicted_indices]

# Print predictions
for sentence, pred_label in zip(test_sentences, predicted_labels):
    print(f"Sentence: {sentence}")
    print(f"Predicted label: {pred_label}")
    print("---")

# Evaluate
print("Accuracy:", accuracy_score(true_labels, predicted_labels))

# Build label list for report
unique_labels = sorted(list(set(true_labels + predicted_labels)))

print(classification_report(true_labels, predicted_labels, labels=unique_labels, target_names=unique_labels))