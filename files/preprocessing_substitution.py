"""
dieses Skript dient zur Durchführung der Hälfte der regulären Datensatzvorverarbeitung
es wird zur Vorverarbeitung der Daten für die TF-IDF-Gewichtung als auch bei zu labelnden Datensätzen eingesetzt
"""

#### ---------- import ----------
import os
import re
import sys
import json
from pathlib import Path


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent
filepath = "{0}\\data\\dataset_processed\\substitution".format(root.absolute())
filecounter = 1

filecount = int(sys.argv[1])
filename_load = "{0}_split".format(sys.argv[2])
filename_target = "{0}_pps".format(sys.argv[2])

# Laden des Datensatzes mit den Akronymen
with open("{0}\\data\\preprocessing_acronymcleanup\\acronyms.jsonl".format(root.absolute())) as file:                   # Datei laden, in der das Dicitionary zu den Akronymen enthalten ist
    acronyms = json.load(file)

# Entfernungen mittels Regular Expressions
regex_sub_format = [                                                                                                    # Liste an Regular-Expressions
    [["<.*?>", "&nbsp", "&ensp", "&emsp"], " "],                                                                        # zur Entfernung von HTML-Tags und HTML-kodierter Leerzeichen
    [[r"[Hh]ttp\S+\s*", r"[Ww]{3}\S+\s*"], " "],                                                                        # zur Entfernung von URLs
    [r"\u2019", "'"]                                                                                                    # zur Enkodierung von Apostrophen zu UTF-8
]

regex_sub_words = [                                                                                                     # Liste mit Regular-Expressions
    [r"\b(\w+)(?:\s+\1\b)+", r"\1"],                                                                                    # zur Reduzierung von ohne Leerzeichen getrennten Wiederholungen von Wörtern (testtesttest)          # diese Pattern können nicht als Gruppe zusammengefasst werden durch Rückreferenzierung
    [r"(\w{2,})(?:\1)+", r"\1"],                                                                                        # zur Reduzierung von mit Leerzeichen getrennten Wiederholungen von Wörtern (test test test)
    [r"(\w)\1{2,}", r"\1"],                                                                                             # zur Reduzierung von hintereinander auftretenden Wiederholungen von Buchstaben in Wörtern, 2x Buchstabe bleibt bestehen (teeeeest)
    [" [Oo]utta ", " out of "],                                                                                         # zur Ersetzung von umgangssprachlichen Abkürzungen
    [" [Gg]otta ", " got "],
    [" [Gg]onna ", " going "],
    [" [Ww]anna ", " want "],
    [" [Ll]emme ", " let me "],
    [" [Gg]imme ", " give me "],
    [" [Dd]unno ", " don't know "],
    [" [Kk]inda ", " quite "],
    [" [Dd]r. ", " doctor "],
    [" [Ww]{3} ", " world wide web "],
    [" [Ss]han't ", " shall not "],
    [" [Ww]on't ", " will not "],
    ["'cause ", " because "],                                                                                           # zur Ersetzung von Kurzformen mit Apostroph in Langformen
    ["'ll ", " will "],
    ["'re ", " are "],
    ["'ve ", " have "],
    ["'m ", " am "],
    [["'d ", "'s "], " "],                                                                                              # Leerzeichen, da 'd für mehrere Wörter stehen könnte (did, would, had, ...)
    [" [Cc]an't ", " can not "],
    ["n't ", " not "]
]

regex_sub_symbols = [                                                                                                   # Liste an Regular-Expressions
    [r"\d", " "],                                                                                                       # zur Entfernung von Zahlen
    [r"[^\w\s]|_", " "],                                                                                                # zur Entfernung von Punkt-/Strichsetzung
    [r"(?<=\w)(?=[A-Z])", " "]                                                                                          # zur Trennung von Wörter am Binnenmajuskel
]

# ---------- functions ----------
# Funktion zum gesammelten Ersetzen aller Regular-Expressions
# vorher werden die Regular-Expressions noch in das richtige Format gebracht
def replace(text, subs):                                                                                                # Funktion führt eine oder mehrere Textersetzungen durch; Substitutions in der Form [[pat1,txt1],[pat2,txt2],...] oder [[[pat1, pat2, ...], sub1], [[pat21, pat22,...], sub2],...]; Rückgabe des resultierenden Textes
    if not isinstance(subs[0], list):                                                                                   # Überprüfung Eingabe
        subs = [subs]                                                                                                   # da nur eine einzelne Substitution vorliegt, muss die Liste zur späteren Iteration einmal genested werden
    else:                                                                                                               # nested Liste; noch unbekannt, ob mehrere Patterns oder mehrere Substitutions
        if len(subs) == 2:                                                                                              # eventuell einzelne Substitution mit mehreren Patterns, andernfalls handelt es sich um eine nested Liste für potenziell mehrere Substitutions
            if not isinstance(subs[1], list):                                                                           # bestätigt [[pat1,pat2,...], txt] als Struktur
                subs = [subs]                                                                                           # da nur eine einzelne Substitution vorliegt, muss die Liste zur späteren Iteration einmal genested werden
    for s in subs:                                                                                                      # alle anderen fehlerhaften Inputs müssen in der Iteration detektiert werden
        if isinstance(s[0], list):
            pat = '|'.join(s[0])                                                                                        # Verbinden mit Regex OR-symbol als Trennzeichen
        else:
            pat = s[0]
        text = re.sub(re.compile(pat), s[1], text)                                                                      # Ersetzen der entsprechenden Zeichenfolgen mittels der Regular-Expressions
    return text


#### ---------- code ----------
while filecounter <= filecount:
    if os.path.exists("{0}\\{1}{2}.jsonl".format(filepath, filename_target, filecounter)):                        # Löschen der bestehenden Dateien; notwendig, da wir nur appenden
        os.remove("{0}\\{1}{2}.jsonl".format(filepath, filename_target, filecounter))
    filecounter += 1


filecounter = 1


while filecounter <= filecount:
    with open("{0}\\data\\dataset_processed\\split\\{1}{2}.jsonl".format(root.absolute(), filename_load, filecounter), "r") as file_input:  # Datei sowie Zeile laden
        for line in file_input:
            json_line = json.loads(line)
            text = json_line["text"]

            # Entfernungen mittels Regular-Expressions
            text = replace(text, regex_sub_format)                                                                      # Ersetzen/Entfernen von: HTML-Tags, HTML-kodierter Leerzeichen, URLs, nicht UTF-8 kodierter Apostrophe

            for acronym, meaning in acronyms.items():                                                                   # Schleife zum Ersetzen von Akronymen mit ihrer Langform, dazu wird eigens angelegter Datensatz verwendet
                pattern_acr = re.compile(" {0} |^{0} | {0}$".format(re.escape(acronym)), re.IGNORECASE)                 # Akronyme können nicht nur zwischen Whitespace stehen, sondern auch am Satzanfang oder -ende
                text = re.sub(pattern_acr, " {0} ".format(meaning), text)                                               # Ersetzen der Akronyme mit ihrer Langform

            text = replace(text, regex_sub_words)                                                                       # Ersetzen/Entfernen/Reduktion von: Wiederholungen (Wörter mit Leerzeichen, Wörter ohne Leerzeichen, mehr als 2x hintereinander auftretende Buchstaben in Wort), Abkürzungen, Kurzformen mit Apostroph
            text = replace(text, regex_sub_symbols)                                                                     # Ersetzen/Entfernen von: Zahlen, Punkt-/Strichsetzung, Binnenmajuskeln

            words = []
            for word in text.split():                                                                                   # Entfernung weiterer Sonderzeichen (nicht ASCII-konforme Wörter)
                if len(word) == len(word.encode()):
                    words.append(word)

            with open("{0}\\{1}{2}.jsonl".format(filepath, filename_target, filecounter), "a+") as file_output:   # Anfügen der bearbeiteten Zeile an Datei
                json.dump({'work_id': json_line["work_id"], 'text': words}, file_output)
                file_output.write('\n')

    filecounter += 1
