"""
dieses Skript dient zur Durchführung der Datensatzvorverarbeitung
dazu gibt es die beiden Funktionen Preprocessing (reguläre Datenvorverarbeitung) und PrePreprocessing (kurze Datenvorverarbeitung vor Datensatzerweiterung)

diesen Code gibt es auch nochmal gesplittet in den Scripts preprocessing_substitution und preprocessing_deletionlemmatization
da für die TF-IDF-Gewichtung eine extra Datenvorverarbeitung notwendig ist, wobei jedoch nur die Hälfte der regulären Datenvorverarbeitung durchlaufen wird
dies ist jedoch insgesamt ablauftechnisch nicht erstrebenswert und sollte zukünftig noch angepasst werden
diese Skripte werden jedoch auch noch für die Datenvorverarbeitung im Falle eines zu labelnden Datensatzes verwendet, da diese selbstständig ausgeführt werden können und nicht wie hier, einzelne Funktionen darstellen
"""

#### ---------- import ----------
import re
import nltk
import json
from pathlib import Path
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent

lemmatizer = WordNetLemmatizer()                                                                                        # Initialisierung Lemmatizer
stopwords = set(nltk.corpus.stopwords.words("english"))                                                                 # Laden der Stopp-Wörter (aus NLTK-Bibliothek)

# Laden des Datensatzes mit den Akronymen
with open("{0}\\data\\preprocessing_acronymcleanup\\acronyms.jsonl".format(root.absolute())) as file:                   # Datei laden, in der das Dicitionary zu den Akronymen enthalten ist
    acronyms = json.load(file)

# Laden des Datensatzes mit den TF-IDF Wörtern
with open("{0}\\data\\tfidf\\tfidf_words.jsonl".format(root.absolute())) as file:                                       # Datei laden, in der die zu entfernenden TF-IDF gewichteten Wörter enthalten sind
    nonecessarywords = json.load(file)

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

regex_augmentation = [                                                                                                  # Liste an Regular-Expressions
    [r"[^\w\s.:!?]", " "],                                                                                              # zur Entfernung von Satzzeichen, aber ohne .!?:
    [" n ", " "],                                                                                                       # zur Entfernung von Zeilenumbrüchen stammenden Überbleibseln
    [r"\s+", " "]                                                                                                       # zur Entfernung von überschüssigen Whitespace
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


# Lemmatizing
#-#-#-#-#
"""
der Code aus diesem Abschnitt (gekennzeichnet mit #-#-#-#-#) ist Fremd-Code
von: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
abgerufen am: 02.12.2023
"""
def get_wordnet_pos(tag):
    tag_dict = {"J": wordnet.ADJ,                                                                                       # Erstellung Dictionary mit POS-Tags
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)                                                                              # Rückgabe Tag; standardmäßige Rückgabe von Substantiv, da POS-Tag evtl. nicht eindeutig zuordnungsbar und somit Substantiv im Normalfall die neutralste Form
#-#-#-#-#


#### ---------- code ----------
def prepreprocessing(text):                                                                                             # kurze Datenvorverarbeitung für Datensatzerweiterung, mittels Regular-Expressions

    text = replace(text, regex_sub_format)                                                                              # Ersetzen/Entfernen von: HTML-Tags, HTML-kodierter Leerzeichen, URLs, nicht UTF-8 kodierter Apostrophe
    text = replace(text, regex_sub_words)                                                                               # Ersetzen/Entfernen/Reduktion von: Wiederholungen (Wörter mit Leerzeichen, Wörter ohne Leerzeichen, mehr als 2x hintereinander auftretende Buchstaben in Wort), Abkürzungen, Kurzformen mit Apostroph
    text = replace(text, regex_augmentation)                                                                            # Ersetzen/Entfernen von: Satzzeichen (außer .!?:), Zeilenumbrüchen, überschüssigem Whitespace

    return text


def preprocessing(text):

    # Entfernungen mittels Regular-Expressions
    text = replace(text, regex_sub_format)                                                                              # Ersetzen/Entfernen von: HTML-Tags, HTML-kodierter Leerzeichen, URLs, nicht UTF-8 kodierter Apostrophe

    for acronym, meaning in acronyms.items():                                                                           # Schleife zum Ersetzen von Akronymen mit ihrer Langform, dazu wird eigens angelegter Datensatz verwendet
        pattern_acr = re.compile(" {0} |^{0} | {0}$".format(re.escape(acronym)), re.IGNORECASE)                         # Akronyme können nicht nur zwischen Whitespace stehen, sondern auch am Satzanfang oder -ende
        text = re.sub(pattern_acr, " {0} ".format(meaning), text)                                                       # Ersetzen der Akronyme mit ihrer Langform

    text = replace(text, regex_sub_words)                                                                               # Ersetzen/Entfernen/Reduktion von: Wiederholungen (Wörter mit Leerzeichen, Wörter ohne Leerzeichen, mehr als 2x hintereinander auftretende Buchstaben in Wort), Abkürzungen, Kurzformen mit Apostroph
    text = replace(text, regex_sub_symbols)                                                                             # Ersetzen/Entfernen von: Zahlen, Punkt-/Strichsetzung, Binnenmajuskeln

    words = []
    for word in text.split():                                                                                           # Entfernung weiterer Sonderzeichen (nicht ASCII-konforme Wörter)
        if len(word) == len(word.encode()):
            words.append(word)

    words = [word for word in words if len(word) > 1 and word not in nonecessarywords and word not in stopwords]        # Entfernung von Wörtern: mit Wortlänge = 1, Stoppwörtern sowie aus der Datei mit den TF-IDF gewichteten Wörtern enthaltene Wörter

    words = [word.lower() for word in words]                                                                            # Umwandlung in Kleinschreibung

    text = ' '.join(words)                                                                                              # Wortliste wird zu String zusammengefügt

    # Lemmatizing
    #-#-#-#-#
    """
    der Code aus diesem Abschnitt (gekennzeichnet mit #-#-#-#-#) ist Fremd-Code
    er wurde jedoch stellenweise noch angepasst und optimiert
    von: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    abgerufen am: 02.12.2023
    """
    words = nltk.word_tokenize(text)                                                                                    # Tokenisierung des Textes
    pos_tags = nltk.pos_tag(words)                                                                                      # POS-Tagging (um Wortart der einzelnen Wörter zu erhalten)
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(tag[0].upper())) for w, tag in pos_tags]                           # Lemmatisierung (vorher Anpassung der Eingabe)
    #-#-#-#-#

    return words
