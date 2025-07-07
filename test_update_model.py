import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved Keras model
model = tf.keras.models.load_model('update_model.h5')

# Load the saved tokenizer
with open('tokenizerupdate.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define intent labels mapping
intent_labels = {
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

def predict_intent(sentence):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
    
    # Predict probabilities for each intent class
    prediction = model.predict(padded_seq)
    
    # Get index of highest probability (predicted intent)
    intent_idx = np.argmax(prediction)
    confidence = prediction[0][intent_idx]
    
    return intent_idx, confidence

if __name__ == "__main__":
    # Example real user queries
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
  "KG extern bei Herrn Fischer heute erfolgt – dokumentieren.",
  "Aktualisiere Mobilitätsfaktor für Frau Klein auf 1,7.",
  "Pflegefachkraft hat Medikamentenkonytrolle bei Herrn Walter gemacht – bitte erfassen.",
  "Füge eine neue Behandlungspflege bei Bewohnerin Schmidt hinzu.",
  "Beratendes Gespräch mit Angehörigen von Frau Brandt eintragen.",
  "Lagerung bei Herrn Dietrich um 9 Uhr durchgeführt – bitte notieren.",
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

    for sentence in test_sentences:
        intent_idx, confidence = predict_intent(sentence)
        if confidence < 0.9:
            print(f"Input: '{sentence}'") 
            print(f"Predicted intent: {intent_labels[intent_idx]} with confidence {confidence:.2f}\n")
