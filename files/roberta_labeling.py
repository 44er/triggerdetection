"""
dieses Skript dient zum Labeln des zu labelnden Datensatzes
dazu werden die gesampelten Dateien geladen und angepasst (z.B. tokenisiert) dem neuronalen Netz übergeben
anschließend wird der Output (Logits) weiterverarbeitet und in die Wort-Form der Labels übersetzt
"""

#### ---------- import ----------
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from datasets import Dataset
from labels import translate_labels
from transformers import AutoTokenizer, RobertaForSequenceClassification


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent
filepath_roberta = "{0}\\data\\roberta".format(root.absolute())
filepath_results = "{0}\\data\\dataset_result".format(root.absolute())
filecounter = 1

filecount = int(sys.argv[1])
filename = sys.argv[2]

# RoBERTa
tokenizer_finetuned = AutoTokenizer.from_pretrained(filepath_roberta)                                                   # Laden von Tokenizer und Modell
model_finetuned = RobertaForSequenceClassification.from_pretrained(filepath_roberta)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")                                     # Verwende wenn möglich GPU anstatt CPU
model_finetuned.to(device)


# ---------- functions ----------
def roberta_labeling(text):                                                                                             # Funktion führt Tokenisierung, Voraussage der Labels mittels RoBERTta und die Umwandlung dieser Ergebnisse in die entsprechenden Wörter durch

    labels_translated = []

    text_tokenized = tokenizer_finetuned(text["text"], truncation=True, padding='max_length', return_tensors="pt", max_length=512, add_special_tokens=True)  # Tokenisierung des übergebenen Textes

    with torch.no_grad():                                                                                               # Labeling mittels des neuronalen Netzes RoBERTa, Output sind zunächst die Logits (Rohwerte ohne Durchlaufen der Aktivierungsfunktion)
        results = model_finetuned(**text_tokenized).logits

    results = nn.functional.softmax(results, dim=1)                                                                     # Anwenden der Transferfunktion Softmax auf die Logit-Werte, welche somit auf Werte zwischen 0 und 1 transferiert werden

    prediction = np.zeros_like(results)                                                                                 # Erstellen eines Arrays mit denselben Dimensionen wie der Output des neuronalen Netzes

    for k in np.ndindex(results.shape):                                                                                 # Es wird ein Array mit 0en und 1en erzeugt, abhängig davon, ob die einzelnen Elemente des Outputs des neuronalen Netzes einen Threshold von 0.2 übersteigen oder nicht
        prediction[k] = results[k] >= 0.2
    prediction = prediction.astype(int)                                                                                 # Umwandlung der Float-Zahlen in Integer

    for book in prediction:                                                                                             # für jedes Buch werden die Labels mittels dem Skript labels.py in Wort-Form umgewandelt
        labels = translate_labels(book.tolist())                                                                        # nicht direkt an Liste anfügen, da von Skript auch Oberkategorie mit zurückgegeben wird
        labels_translated.append(labels[0])                                                                             # nur die Unterkategorien werden übernommen

    return {'labels': labels_translated}                                                                                # Rückgabe der Ergebnisse als Dictionary


#### ---------- code ----------
if os.path.exists("{0}\\{1}_labeled.jsonl".format(filepath_results, filename)):                                   # Löschen der bestehenden Datei; notwendig, da wir nur appenden
    os.remove("{0}\\{1}_labeled.jsonl".format(filepath_results, filename))


while filecounter <= filecount:

    with open("{0}\\data\\dataset_sampled\\{1}_sampled{2}.jsonl".format(root.absolute(), filename, filecounter)) as file_input:    # Einlesen der gesampelten Dateien

        workids = []
        text = []
        lines = {}

        for line in file_input:                                                                                         # Zeileninhalt einlesen
            json_line = json.loads(line)
            workids.append(json_line["work_id"])
            text.append(' '.join(json_line["text"]))                                                                    # Wortliste zu String zusammenfügen

        lines["workid"] = workids                                                                                       # Transformation des eingelesenen Zeileninhalts zu Datensatz (hier: Dictionary, Zeile -> Liste -> Dictionary -> DataFrame -> Datensatz)
        lines["text"] = text
        df = pd.DataFrame(lines)
        dataset_orig = Dataset.from_pandas(df)

        dataset_labeled = dataset_orig.map(roberta_labeling, batched=True)                                              # komplettes Verarbeiten des Datensatzes in der Funktion roberta_labeling, Zurückgegeben wird ein fertig gelabelter Datensatz

        for book in dataset_labeled:                                                                                    # Speichern der Ergebnisse als Jsonl-Datei
            with open("{0}\\{1}_labeled.jsonl".format(filepath_results, filename), "a+") as file_output:
                json.dump({'work_id': book["workid"], 'labels': book["labels"]}, file_output)
                file_output.write('\n')

    filecounter += 1
