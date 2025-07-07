import speech_recognition as sr
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("intent_model.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

labels = [
    "Show Data", "Update Data", "Insert Data",
    "Analyze Data", "Events/Settings", "Casual Talk"
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