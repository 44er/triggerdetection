"""
dieses Skript dient zur Datensatzerweiterung (Augmentation)
das Sampling-Skript ruft gegebenenfalls die Funktion augmentation auf

In der Funktion Augmentation wird dann zufällig die auszuführende Datensatzerweiterung-Funktion bestimmt
mit einer Wahrscheinlichkeit von 20 % wird im Anschluss erneut eine zufällige Datensatzerweiterung-Funktion auf den Datensatz angewendet

Datensatzerweiterung-Funktionen:
- keine Veränderung
- Synonym-Ersetzung
- kontextbehaftetes Worteinfügen
- kontextbehaftetes Wortlöschen
- zufälliges Vertauschen von Wörtern
- zufälliges Löschen von Wörtern
- Backtranslation mit Spanisch
- Backtranslation mit Russisch
- Backtranslation mit Chinesisch

die Funktionen sind von nlpaug und HuggingFace
je nach Funktion muss der Input noch angepasst werden
"""

#### ---------- import ----------
import re
import math
import random
import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer


#### ---------- definition ----------

# ---------- functions ----------
def splitting(text):                                                                                                    # Funktion zum Splitten des Inputs an Satzeichen (.?!:)
    split_sentences = re.split(r'[.?!:]', text)
    chunks = [sentence.strip() for sentence in split_sentences if sentence.strip()]
    return chunks


def joining(chunks):                                                                                                    # Funktion zum Verbinden der Chunks im Input zu String
    text = [''.join(e) for e in chunks]
    text = '. '.join(text)                                                                                              # mit Punkt verbinden, damit bei weiterer Augmentation-Funktion Satztrennung erfolgen kann bzw. bei Backtranslation notwendig
    return text


def donothing(text, min, max):                                                                                          # keine Veränderung
    return text


#---- Synonym-Ersetzung
def synonym_replacement(text, min, max):                                                                                # Ersetzen von Wörtern durch ihre Synonyme
    aug = naw.SynonymAug(aug_src='wordnet', lang='eng', aug_min=min, aug_max=max)
    return aug.augment(text)


#---- Word Embedding
def adjustment(text, funcs, min, max, func):                                                                            # Funktion zum Anpassen des Inputs sowie Nachbearbeitung des Outputs für die Funktionen wordembedding_insert & wordembedding_substitute
    chunks_processed = []
    chunks = splitting(text)                                                                                            # Sätze werden gesplittet und anschließend einzeln bearbeitet (da ansonsten Probleme mit verwertbaren Output)
    min = math.ceil((min / len(chunks))/2)                                                                              # Berechnung von min und max für jeden Satz; warum nochmal :2 und nicht min/len(chunks)? Somit mehr Variation und Zufall (durch Wordembedding-Funktion selbst), da Sätze unterschiedlich lang
    max = math.ceil(max / len(chunks))
    for chunk in chunks:
        text = funcs[func](chunk, min, max)                                                                             # Wordembedding-Funktion mit Satz ausführen
        if isinstance(text, list):                                                                                      # Wordembedding-Funktion erzeugt immer nochmal eigene Liste, somit ansonsten Liste in Liste
            text = text[0]
        chunks_processed.append(text)
    text = joining(chunks_processed)                                                                                    # Chunks wieder zu String zusammenfügen
    return text

def wordembedding_insert(text, min, max):                                                                               # kontextbehaftetes Einfügen von Wörtern
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="insert", batch_size=32, aug_min=min, aug_max=max)
    return aug.augment(text)


def wordembedding_substitute(text, min, max):                                                                           # kontextbehaftetes Löschen von Wörtern
    aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action="substitute", batch_size=32, aug_min=min, aug_max=max)
    return aug.augment(text)


#---- Zufall
def random_swap(text, min, max):                                                                                        # zufälliger Tausch von Wörtern
    aug = naw.RandomWordAug(action="swap", aug_min=min, aug_max=max)
    return aug.augment(text)


def random_delete(text, min, max):                                                                                      # zufälliges Löschen von Wörtern
    aug = naw.RandomWordAug(action="delete", aug_min=min, aug_max=max)
    return aug.augment(text)


#---- Backtranslation
def translation(text, model, tokenizer, language):                                                                      # Funktion zum Anpassen des Inputs der Backtranslations sowie Durchführung dieser
    chunks_translated = []
    chunks = splitting(text)                                                                                            # Sätze werden gesplittet und anschließend einzeln bearbeitet (da ansonsten Probleme mit verwertbaren Output)
    for chunk in chunks:
        if language != "en":
            chunk = ">>{}<< {}".format(language, chunk)                                                           # notwendige Formatierung für Modell
        translate = model.generate(**tokenizer(chunk, return_tensors="pt", padding=True, truncation=True))              # Tokenisierung und Übersetzung mittels HuggingFace-Modelle     # diese Code-Zeile wurde von https://www.kaggle.com/code/keitazoumana/data-augmentation-in-nlp-with-back-translation übernommen, 22.12.2023
        text_translated = [tokenizer.decode(t, skip_special_tokens=True) for t in translate]                            # Dekodierung des übersetzten Textes                            # diese Code-Zeile wurde von https://www.kaggle.com/code/keitazoumana/data-augmentation-in-nlp-with-back-translation übernommen, 22.12.2023
        chunks_translated.append(text_translated)
    transtext = joining(chunks_translated)                                                                              # Chunks wieder zu String zusammenfügen
    return transtext


def backtranslation_alllanguages(targetlanguage, newlanguage, originallanguage, text):                                  # Funktion zum Laden der Datensätze/Modelle zu den übergebenen Sprachen
    mnl_tokenizer = MarianTokenizer.from_pretrained(newlanguage)
    mol_tokenizer = MarianTokenizer.from_pretrained(originallanguage)
    mnl = MarianMTModel.from_pretrained(newlanguage)
    mol = MarianMTModel.from_pretrained(originallanguage)
    translation_newlanguage = translation(text, mnl, mnl_tokenizer, targetlanguage)                                     # Übersetzung in Fremdsprache
    translation_originallanguage = translation(translation_newlanguage, mol, mol_tokenizer, 'en')               # Rückübersetzung in Englisch
    return translation_originallanguage


def backtranslation_spanish(text, min, max):                                                                            # Backtranslation mit Englisch & Spanisch
    newlanguage = 'Helsinki-NLP/opus-mt-en-es'                                                                          # Laden der Namen der Datensätze
    originallanguage = 'Helsinki-NLP/opus-mt-es-en'
    translation = backtranslation_alllanguages('es', newlanguage, originallanguage, text)
    return translation


def backtranslation_russian(text, min, max):                                                                            # Backtranslation mit Englisch & Russisch
    newlanguage = 'Helsinki-NLP/opus-mt-en-ru'                                                                          # Laden der Namen der Datensätze
    originallanguage = 'Helsinki-NLP/opus-mt-ru-en'
    translation = backtranslation_alllanguages('ru', newlanguage, originallanguage, text)
    return translation


def backtranslation_chinese(text, min, max):                                                                            # Backtranslation mit Englisch & Chinesisch
    newlanguage = 'Helsinki-NLP/opus-mt-en-zh'                                                                          # Laden der Namen der Datensätze
    originallanguage = 'Helsinki-NLP/opus-mt-zh-en'
    translation = backtranslation_alllanguages('zh', newlanguage, originallanguage, text)
    return translation


#### ---------- code ----------
def augmentation(text):

    wordcounter = len(text.split())                                                                                     # Bestimmung der Länge des Inputs
    min = int(wordcounter * 0.1)                                                                                        # min. 10%, max. 25% der Wörter (manche Datensatzerweiterungs-Funktionen benötigen ein Minimum und Maximum, in welchem sie dann zufällig die Anzahl der zu verarbeitenden Wörter bestimmen)
    max = int(wordcounter * 0.25)
    funcs = [donothing, synonym_replacement, wordembedding_insert, wordembedding_substitute, random_swap, random_delete, backtranslation_spanish, backtranslation_russian, backtranslation_chinese]  # mögliche Datensatzerweiterung-Funktionen

    func1 = int(random.uniform(0, len(funcs)))                                                                       # Zufallszahl, welche Funktion anschließend ausgeführt wird
    chance = random.uniform(0, 1)                                                                                 # zufällige Prozentzahl
    if chance > 0.8:                                                                                                    # mit einer WSK von 20 % wird im Anschluss direkt eine zweite Datensatzerweiterungs-Funktion angewendet
        func2 = int(random.uniform(0, len(funcs)))                                                                   # Zufallszahl, welche Funktion anschließend ausgeführt wird
    else:
        func2 = 0                                                                                                       # gleichbedeutend mit Funktion donothing

    if func1 in {2, 3}:                                                                                                 # ist Funktion = Wordembedding? dann Anpassung der Eingabe
        text = adjustment(text, funcs, min, max, func1)
    else:
        text = funcs[func1](text, min, max)
    if isinstance(text, list):                                                                                          # Rückgabe ist je nach Funktion eine Liste mit einem Element, wir benötigen jedoch einen String
        text = text[0]

    if func2 in {2, 3}:                                                                                                 # ist Funktion = Wordembedding? dann Anpassung der Eingabe
        text = adjustment(text, funcs, min, max, func2)
    else:
        text = funcs[func2](text, min, max)
    if isinstance(text, list):                                                                                          # Rückgabe ist je nach Funktion eine Liste mit einem Element, wir benötigen jedoch einen String
        text = text[0]

    return text
