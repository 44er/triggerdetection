"""
dieses Skript fine-tuned das vortrainierte neuronale Netz RoBERTa von HuggingFace mit dem gesampelten Datensatz
anschließend kann dieses zum Labeln von Datensätzen verwendet werden

es wird der Trainingsalgorithmus von HuggingFace verwendet
somit stammen die Funktionen wie z.B. TrainingArguments und Trainer von HuggingFace (Code beispielsweise von https://huggingface.co/docs/transformers/training), werden aber nicht extra als Fremd-Code gekennzeichnet, da es sich um Funktionen handelt
der Vorteil der vorgefertigen Funktionen ist, dass viele Sachen, wie beispielsweise die Anpassung der Parameter oder die endgültige Evaluation des Modells, komplett automatisch ablaufen
als Tokenizer und Modell wurden die Standard-Funktionen von RoBERTa-Base gewählt
die genauen Hyperparameter können bei TrainingArguments eingesehen werden

folgende Funktionen werden verwendet:
- Aktivierungsfunktion: GELU
- Loss-Funktion: BCE Loss
- Optimizer: Adam

da die Konfigurationsdateien bei Ausführung des Codes automatisch heruntergeladen werden, werden diese nicht mit zur Verfügung gestellt


!!!! AKTUELL IST DIE K-FOLD CROSS-VALIDATION FALSCH EINGEBUNDEN
"""

#### ---------- import ----------
import sys
import json
import torch
import wandb
import evaluate
import numpy as np
import pandas as pd
from torch import nn
from pathlib import Path
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
)
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent
filepath = "{0}\\data\\roberta".format(root.absolute())
filecounter = 1

filecount = int(sys.argv[1])
filename_load = "{0}_sampled".format(sys.argv[2])

# RoBERTa                                                                                                               # RoBERTa-Konfiguration
modelname = "roberta-base"                                                                                              # als Modell wird RoBERTa-base verwendet
tokenizer = AutoTokenizer.from_pretrained(modelname, padding=True, truncation=True)                                     # Initialisierung des Tokenizers
model = RobertaForSequenceClassification.from_pretrained(modelname, num_labels=32, problem_type="multi_label_classification")   # Initialisierung des Modells
config = AutoConfig.from_pretrained(modelname)                                                                          # Laden der Konfiguration von RoBERTa-base
config.hidden_dropout_prob = 0.2                                                                                        # Anpassung der Dropout-Werte
config.attention_probs_dropout_prob = 0.2

skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=44)                                              # Implementation einer stratified 5-fold Cross-Validation für Multiclassification-Probleme

wandb.login(key="")                                                                                                     # der HuggingFace-Trainer nutzt standardmäßig die Website "Weight & Biases", auf welcher nach dem Training der Verlauf und die Auswirkung verschiedener Parameter eingesehen werden kann
                                                                                                                        # an dieser Stelle muss noch der zum privaten Account gehörende Key eingefügt werden!

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")                                     # Verwende wenn möglich GPU anstatt CPU
model.to(device)


# ---------- functions ----------
def tokenization(text):
    return tokenizer(text["text"], truncation=True, padding='max_length', return_tensors="pt", max_length=512, add_special_tokens=True)     # Funktion zur Tokenisierung des Inputs


def compute_metrics(modellprediction):                                                                                  # Funktion zum Berechnen der verschiedenen Metriken
    logits, labels = modellprediction                                                                                   # Übergabe der Parameter logits (Rohwerte, = Ausgabe von neuronalen Netz) und der Labels
    logits = torch.tensor(logits)                                                                                       # Logits werden bei nachfolgendem Schritt jedoch als Tensor-Vektor benötigt
    logits = nn.functional.softmax(logits, dim=1)                                                                       # Anwenden der Transferfunktion Softmax auf die Logit-Werte, welche somit auf Werte zwischen 0 und 1 transferiert werden
    prediction = np.zeros_like(logits)                                                                                  # Erstellen eines Arrays mit denselben Dimensionen wie der Output des neuronalen Netzes
    for k in np.ndindex(logits.shape):                                                                                  # Es wird ein Array mit 0en und 1en erzeugt, abhängig davon, ob die einzelnen Elemente des Outputs des neuronalen Netzes einen Threshold von 0.2 übersteigen oder nicht
        prediction[k] = float(logits[k] >= 0.2)
    precision = precision_score(y_true=labels, y_pred=prediction, average='micro')                                      # Berechnung der Precision
    recall = recall_score(y_true=labels, y_pred=prediction, average='micro')                                            # Berechnung des Recalls
    f1 = f1_score(y_true=labels, y_pred=prediction, average='micro')                                                    # Berechnung des F1-Maßes
    accuracy = accuracy_score(y_true=labels, y_pred=prediction)                                                         # Berechnung der Accuracy
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}                                   # Rückgabe der Scores                       # diese Code-Zeile wurde von https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b übernommen, 30.11.2023


training_args = TrainingArguments(                                                                                      # Konfiguration der Trainingsparameter für den HuggingFace-Trainer; aktuell auf die bisher am besten funktionierenden Werte gestellt
    output_dir="{0}\\results".format(filepath),
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_accumulation_steps=100,
    learning_rate=1e-4,
    weight_decay=0.02,
    max_grad_norm=1.0,
    num_train_epochs=50,
    lr_scheduler_type="linear",
    warmup_steps=100,
    logging_dir='{0}\\logs'.format(filepath),
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=5,
    seed=44,
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    optim="adamw_torch",
    push_to_hub=False,
)


#### ---------- code ----------
while filecounter <= filecount:

    with open("{0}\\data\\dataset_sampled\\{1}{2}.jsonl".format(root.absolute(), filename_load, filecounter)) as file_input:    # Laden der gesampelten Dateien

        labels = []
        text = []
        lines = {}

        for line in file_input:
            json_line = json.loads(line)                                                                                # Zeileninhalt einlesen
            labels.append([float(i) for i in json_line["labels"]])                                                      # Umwandlung von Integer zu Float, da so von neuronalem Netz erwartet
            text.append(' '.join(json_line['text']))                                                                    # Wortliste zu String zusammenfügen
        lines["text"] = text                                                                                            # Transformation des eingelesenen Zeileninhalts zu Datensatz (hier erstmal zu Dictionary; Zeile -> Liste -> Dictionary -> DataFrame)
        lines["labels"] = labels

    df = pd.DataFrame(lines)

    kfold_text = df["text"].to_numpy()                                                                                  # Anpassen des DataFrames für die k-Fold Cross-Validation
    kfold_labels = np.vstack(df["labels"])

    for fold, (train_index, valid_index) in enumerate(skf.split(kfold_text, kfold_labels)):                             # Splitten des Inputs in eine stratified 5-fold Cross-Validation und iterieren über diese Chunks; die zu vergleichenden Scores müssen aktuell noch per Hand verglichen werden

        dataset_orig = DatasetDict({                                                                                    # Erstellen eines Datensatzes anhand der Splittung durch die 5-fold Cross-Validation und des DataFrames
            'train': Dataset.from_pandas(df.iloc[train_index]),
            'validation': Dataset.from_pandas(df.iloc[valid_index]),
        })
        dataset_orig = dataset_orig.remove_columns(["__index_level_0__"])                                               # Entfernen der Spalte Index, welche bei der Konvertierung von DataFrame zu Dataset automatisch übernommen wird

        dataset_tokenized = dataset_orig.map(tokenization, batched=True)                                                # Tokenisierung des Datensatzes mit dem Tokenizer von RoBERTa

        trainer = Trainer(                                                                                              # Konfiguration des HuggingFace-Trainers
            model=model,
            args=training_args,                                                                                         # Hyperparameter
            train_dataset=dataset_tokenized["train"],                                                                   # Trainingsdatensatz
            eval_dataset=dataset_tokenized["validation"],                                                               # Validierungsdatensatz
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt"),         # Anpassung der Batches durch Collator
            compute_metrics=compute_metrics,                                                                            # Metriken
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]                                                # Funktion zum Vermeiden von Overfitting
        )

        trainer.train()                                                                                                 # Beginnen des Trainings; die engültige Evaluation erfolgt automatisch durch den HuggingFace-Trainer

    filecounter += 1

model.save_pretrained(filepath)                                                                                         # Speichern des trainierten neuronalen Netzes
tokenizer.save_pretrained(filepath)                                                                                     # Speichern des Tokenizers
