import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved Keras model
model = tf.keras.models.load_model('intent_model.h5')

# Load the saved tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define intent labels mapping
intent_labels = {
    0: "Zeige Daten",
    1: "Aktualisiere Daten",
    2: "Erstelle Daten",
    3: "Analisiere Daten",
    4: "Events oder Einstellungen",
    5: "Small Talk"
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
        "Kannst du mir die Daten von Frau Meier zeigen?",
    "Ich will wissen, was du über Herr Weber gespeichert hast",
    "Zeig mir den Datensatz von Anna",
    "Was weißt du über Peter Müller?",
    "Lade bitte die Akte von Klaus",
    "Ich möchte auf die Informationen zu Lena zugreifen",
    "Zeige mir die gespeicherten Informationen von Tobias",
    "Welche Daten hast du über Frau Schmitt?",
    "Ändere bitte die Telefonnummer von Herr Krause",
    "Ich will den Wohnort von Frau Lehmann aktualisieren",
    "Bitte ergänze neue Werte zu den Daten von Max",
    "Passe die Angaben von Jana an",
    "Korrigiere die Adresse von Frau Schulz",
    "Aktualisiere bitte den Datensatz von Tim",
    "Bearbeite die Patientendaten von Sabine",
    "Ich muss die Angaben zu Nico ändern",
    "Erstelle einen neuen Eintrag für Frau König",
    "Ich möchte neue Informationen zu Sven hinzufügen",
    "Trage bitte einen neuen Patienten namens Paul ein",
    "Füge eine neue Akte für Lisa hinzu",
    "Neuen Datensatz für Sarah erstellen",
    "Bitte registriere Laura im System",
    "Ein neuer Fall für Herr Mayer soll angelegt werden",
    "Füge einen neuen Kontakt zu deinem System hinzu",
    "Lege bitte eine neue Person mit dem Namen Emma an",
    "Analysiere bitte die Daten von Frau Berger",
    "Ich brauche eine Auswertung für Tom",
    "Zeige mir die Analyse von Herr Richter",
    "Wie sehen die Daten von Maria in der Zusammenfassung aus?",
    "Was kannst du mir zur Entwicklung von Jonas sagen?",
    "Gib mir einen Bericht zu Frau Hoffmann",
    "Analysiere bitte die Gesundheitsdaten von Tim",
    "Welche Muster erkennst du bei Laura?",
    "Welche Termine hat Leon diese Woche?",
    "Was steht bei Jonas im Kalender?",
    "Zeig mir die Einstellungen von Jana",
    "Was ist der nächste geplante Termin von Sarah?",
    "Welche Ereignisse sind für Paul geplant?",
    "Wann ist der nächste Kontrolltermin von Emma?",
    "Welche Systemeinstellungen hat Herr Becker?",
    "Was ist bei Max für heute vorgesehen?",
    "Wie geht es dir heute?",
    "Kannst du mir einen lustigen Witz erzählen?",
    "Was kannst du alles tun?",
    "Was ist deine Lieblingsfarbe?",
    "Wie ist das Wetter draußen?",
    "Hast du einen Namen?",
    "Was ist dein Zweck?",
    "Bist du ein echter Mensch?",
    "Erzähl mir was über dich",
    "Ich finde dich interessant"
    ]

    for sentence in test_sentences:
        intent_idx, confidence = predict_intent(sentence)
        print(f"Input: '{sentence}'")
        print(f"Predicted intent: {intent_labels[intent_idx]} with confidence {confidence:.2f}\n")
