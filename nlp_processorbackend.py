# nlp_processor.py
import spacy
import sys
import json

nlp = spacy.load("en_core_web_sm")

# Get input sentence from PHP
#input_text = sys.argv[1]
input_text = "Medibot, update Markus Berger sein Blutdruck"

#doc = nlp(input_text)
doc = nlp("Medibot, update Markus Berger sein Blutdruck")

# Extract PERSONs
persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

# Heuristics for fields
fields = []
possible_fields = ["blut", "pulse", "medikament", "temperatur", "termine", "informationen"]
for field in possible_fields:
    if field in input_text.lower():
        fields.append(field)

# Heuristics for intent
if "update" in input_text.lower():
    intent = "update"
elif "add" in input_text.lower() or "insert" in input_text.lower():
    intent = "insert"
elif "show" in input_text.lower() or "check" in input_text.lower():
    intent = "show"
elif "analyze" in input_text.lower():
    intent = "analyze"
elif "appointment" in input_text.lower() or "schedule" in input_text.lower():
    intent = "event"
else:
    intent = "chat"

output = {
    "intent": intent,
    "persons": persons,
    "fields": fields
}

print(json.dumps(output))