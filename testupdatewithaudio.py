import speech_recognition as sr
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("update_model.h5")

with open("tokenizerupdate.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

labels = [
    "update_blutzucker",
 "update_blutdruck",
 "update_temperatur",
 "update_gewicht",
 "update_medikation",
 "update_ein_ausfuhr",
 "update_stuhlgang",
 "update_einzelbetreuung",
 "update_verbandswechsel ",
 "update_medikamenten_stellung",
 "update_nahrungsaufnahme",
 "update_fluessigkeitsaufnahme",
 "update_fixierung_bett",
 "update_remove_fixierung_bauchgurt",
 "update_gruppenbetreuung",
 "update_tagesgruppe",
 "update_medikamente_vorbereiten",
 "update_fixierung_rollstuhl",
 "update_remove_fixierung_rollstuhl",
 "update_fixierung_therapietisch",
 "update_remove_fixierung_therapietisch",
 "update_betreuung_43b",
 "update_behandlungspflege",
 "update_medikamente_bereitstellen",
 "update_beratende_gespraeche",
 "update_kg_extern",
 "update_mobilitaetsfaktor",
 "update_validierende_gespraeche",
 "update_lagerungsprotokoll",
 "update_duschen",
 "update_frisoer",
 "update_fusspflege",
 "update_nagelpflege",
 "update_haarwaesche",
 "update_fixierung_rollstuhlbremse_anlegen",
 "update_remove_fixierung_rollstuhlbremse",
 "update_toilettengang",
 "update_medikamentenkontrolle",
 "update_betreuung_beobachtung",
 "update_fixierung",
 "update_gewichtsauswertung",
 "update_medikamenten_vorbereitung_diabetiker",
 "update_medikamente_verabreichen_pfk"
]

# Listen from mic
def listen_and_classify():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("\n🎙️ Please speak now...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="de-DE")
        print(f"🗣️ You said: {text}")

        # Preprocess
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=8, padding='post')

        # Predict
        predictions = model.predict(padded)
        confidence = np.max(predictions)
        intent_idx = np.argmax(predictions)
        intent = labels[intent_idx]

        print(f"🤖 Predicted intent: {intent} (Confidence: {confidence:.2f})")

    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError as e:
        print(f"❌ Error with the speech recognition service: {e}")

# Run it
while True:
    listen_and_classify()
    if input("\nDo you want to test again? (y/n): ").lower() != "y":
        break