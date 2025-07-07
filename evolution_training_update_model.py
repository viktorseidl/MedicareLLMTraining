import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from deap import base, creator, tools, algorithms
import random
import tensorflow as tf

#<Name>

# Example training data
sentences = [
    "Bitte aktualisiere den Blutzuckerwert auf 125 mg/dl bei Frau Müller.",  # 0
    "Trage die Temperatur von Herrn Schmidt mit 37,8 Grad ein.",  # 2
    "Aktualisiere das Gewicht von Frau Lehmann auf 68 Kilogramm.",  # 3
    "Dokumentiere den Verbandswechsel bei Herrn Braun heute um 10 Uhr.",  # 8
    "Notiere, dass Frau Schulz ihre Medikamente um 14 Uhr erhalten hat.",  # 4
    "Pflegefachkraft hat die Medikamente bei Herrn Meier verabreicht – bitte eintragen.",  # 42
    "Füge einen Termin zur Einzelbetreuung für Frau König um 15 Uhr hinzu.",  # 7
    "Ergänze, dass die Gruppentherapie mit Bewohnern aus Wohnbereich 2 stattgefunden hat.",  # 14
    "Fixierung im Rollstuhl bei Herrn Neumann wurde angelegt – bitte dokumentieren.",  # 17
    "Bitte trage die Medikamentenvorbereitung für Diabetikerin Frau Schulze ein.",  # 41
    "Ein- und Ausfuhr bei Herrn Weber: Einfuhr 1200 ml, Ausfuhr 1100 ml.",  # 5
    "Eintrag: Fixierung am Therapietisch bei Frau Krüger durchgeführt.",  # 19
    "Update: Bewohnerin Frau Wagner wurde heute geduscht.",  # 29
    "Nagelpflege bei Herrn Schröder durchgeführt – bitte notieren.",  # 32
    "Fußpflege bei Frau Hartmann abgeschlossen – trage es ein.",  # 31
    "Temperatur bei Herrn Schwarz auf 38,2 aktualisieren.",  # 2
    "Verabreichung von Insulin bei Frau Becker dokumentieren.",  # 4
    "Fixierung mit Bauchgurt im Bett bei Herrn Kunze aktiv – bitte eintragen.",  # 12
    "Fixierung Bauchgurt im Rollstuhl wurde entfernt bei Frau Peters.",  # 13
    "Trage validierendes Gespräch mit Frau Nowak heute um 11:30 Uhr ein.",  # 27
    "Medikamente für Frau Jansen wurden heute gestellt – bitte eintragen.",  # 9
    "Die KG wurde extern bei Herrn Fischer heute erledigt – dokumentieren.",  # 25
    "Aktualisiere Mobilitätsfaktor für Frau Klein auf 1,7.",  # 26
    "Unsere Pflegefachkraft hat die Medikamentenkontrolle bei Herrn Walter durchgeführt.",  # 37
    "Füge eine neue Behandlungspflege bei Bewohnerin Schmidt hinzu.",  # 22
    "Beratendes Gespräch mit Angehörigen von Frau Brandt eintragen.",  # 24
    "Lagerung bei Herrn Dietrich wurde um 9 Uhr durchgeführt – bitte notieren.",  # 28
    "Eintrag: Nahrungsaufnahme bei Frau Lange normal.",  # 10
    "Einzelbetreuung bei Frau Keller von 13 bis 14 Uhr dokumentieren.",  # 7
    "Friseurtermin bei Herrn Ludwig heute abgeschlossen – bitte notieren.",  # 30
    "Neue Fixierung am Therapietisch bei Frau Vogt aktiviert.",  # 19
    "Bereite Medikamente für Herrn Scholz vor – eintragen.",  # 16
    "Aktualisiere Eintrag: Medikamente verabreicht bei Frau Bergmann um 16 Uhr.",  # 4
    "Einfuhr 1500 ml bei Herrn Otto – bitte dokumentieren.",  # 5
    "Update: Bauchgurt im Bett bei Frau Seidel wurde entfernt.",  # 13
    "Dokumentiere bitte, dass die Gruppenbetreuung heute stattgefunden hat.",  # 14
    "Medikamente bei Frau Keller vorbereitet – bitte im System speichern.",  # 16
    "Duschen bei Herrn Pfeiffer abgeschlossen – eintragen.",  # 29
    "Fixierung an Bauchgurt im Rollstuhl bei Frau Frank erneut angelegt.",  # 17
    "Trage ein: Medikamentengabe bei Herrn Paulsen durch PFK erfolgt."  # 42
]

labels_text = [
    0, 2, 3, 8, 4, 42, 7, 14, 17, 41,
    5, 19, 29, 32, 31, 2, 4, 12, 13, 27,
    9, 25, 26, 37, 22, 24, 28, 10, 7, 30,
    19, 16, 4, 5, 13, 14, 16, 29, 17, 42
]
# Label encoding
label_map = {label: i for i, label in enumerate(sorted(set(labels_text)))}
labels = np.array([label_map[label] for label in labels_text])

# Tokenize sentences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_map)
print("📐 Padded shape:", padded.shape)
# -------------------------------
# ✅ Step 2: Model Architecture
# -------------------------------
def create_model():
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=padded.shape[1]),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# ✅ Step 3: DEAP Evolution Setup
# -------------------------------
def evaluate(individual):
    model = create_model()
    weights = model.get_weights()
    idx = 0
    new_weights = []
    for w in weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(np.array(individual[idx:idx + size]).reshape(shape))
        idx += size

    model.set_weights(new_weights)
    loss, acc = model.evaluate(padded, labels, verbose=0)
    return acc,

def get_model_size():
    model = create_model()
    
    model.build(input_shape=(None, padded.shape[1]))  # Force build
    return sum(np.prod(w.shape) for w in model.get_weights())

# Setup DEAP 
weight_size = get_model_size() 
print("🧠 Model genome size:", weight_size)
if weight_size < 2:
    raise ValueError(f"Model weight genome too small for crossover (size: {weight_size}). Increase model size.")
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, weight_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# -------------------------------
# ✅ Step 4: Run the Evolution
# -------------------------------
def run_evolution():
    pop = toolbox.population(n=20)
    NGEN = 10  # For better results, use 50-100

    for gen in range(NGEN):
        print(f"🔄 Generation {gen}")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop = toolbox.select(offspring, k=len(pop))

    best = tools.selBest(pop, k=1)[0]
    return best

# -------------------------------
# ✅ Step 5: Save Best Model
# -------------------------------
def save_best_model(best_genome):
    model = create_model()
    weights = model.get_weights()
    idx = 0
    new_weights = []
    for w in weights:
        shape = w.shape
        size = np.prod(shape)
        new_weights.append(np.array(best_genome[idx:idx + size]).reshape(shape))
        idx += size
    model.set_weights(new_weights)
    model.save("best_model.h5") 
    print("✅ Best model saved to 'best_model.h5'.")
import pickle 
with open('tokenizerupdate.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
# -------------------------------
# 🚀 Main Entry Point
# -------------------------------
if __name__ == "__main__":
    best = run_evolution()
    save_best_model(best)