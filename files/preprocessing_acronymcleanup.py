"""
dieses Skript dient dazu, Tabellen von zwei heruntergeladenen Webseiten auszuwerten und eine Jsonl-Datei mit einem Dictionary zu Akronymen und deren Langform zu erzeugen
durch das Dictionary werden Dopplungen der Akronyme unter den Webseiten automatisch entfernt

Webseiten: gedownloaded am 29.11.2023
https://www.muller-godschalk.com/acronyms.html
https://www.sprachheld.de/englische-abkuerzungen/
die Webseiten wurden händisch leicht verändert und an die Bedürfnisse angepasst
"""

#### ---------- import ----------
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup


#### ---------- definition ----------
path = Path(__file__)
root = (path.parent).parent

pattern_strip = re.compile('[ \n]+')                                                                                    # Pattern, welches Zeilenumbrüche detektiert
acronyms = {}

# ---------- functions ----------
def strip(s):                                                                                                           # Funktion zum Entfernen von Zeilenumrbüchen
    return re.sub(pattern_strip, ' ', s)


#### ---------- code ----------

# ---------- gespeicherte HTML-Seiten einlesen ----------
dataset1 = open("{0}\\data\\preprocessing_acronymcleanup\\website1_mullergodschalk.htm".format(root.absolute()))
dataset2 = open("{0}\\data\\preprocessing_acronymcleanup\\website2_sprachheld.htm".format(root.absolute()))

# ---------- HTML Vorverarbeitung ----------
dataset1_clean = BeautifulSoup(dataset1, "html.parser")
dataset2_clean = BeautifulSoup(dataset2, "html.parser")

# ---------- Auswertung Tabelle ----------
dataset1_acronyms = dataset1_clean.find_all("tr")                                                                       # Zeilen mit dem HTML-Tags "tr" finden
dataset2_acronyms = dataset2_clean.find_all("tr")


# ---------- Verarbeiten erster Website ----------
for row in dataset1_acronyms:
    dataset1_columns = row.find_all("td")                                                                               # Spalten anhand des HTML-Tags "td" finden
    if len(dataset1_columns) == 2:                                                                                      # Test, ob es exakt zwei Spalten sind (Akronym & Bedeutung)
        acronym = dataset1_columns[0].text.strip()                                                                      # wenn ja, dann Inhalt auslesen
        meaning = strip(dataset1_columns[1].text)                                                                       # und Entnfernen von Zeilenumbrüchen
        if acronym and meaning:                                                                                         # Test, ob Akronym und Bedeutung vorhanden sind (da teilweise leere Tabellenspalten)
            acronyms[acronym] = meaning


# ---------- Verarbeiten zweiter Website ----------
for i, row in enumerate(dataset2_acronyms):
    if i == 0:                                                                                                          # Ignorieren des Tabellenkopfes
        continue
    dataset2_columns = row.find_all("td")                                                                               # Spalten anhand des HTML-Tags "td" finden
    if len(dataset2_columns) == 3:                                                                                      # Test, ob es exakt drei Spalten sind (Akronym & Bedeutung & Übersetzung)
        acronym = dataset2_columns[0].text.strip()                                                                      # wenn ja, dann Inhalt auslesen
        meaning = strip(dataset2_columns[1].text)                                                                       # und Entnfernen von Zeilenumbrüchen
        if acronym and meaning:                                                                                         # Test, ob Akronym und Bedeutung vorhanden sind (da teilweise leere Tabellenspalten)
            acronyms[acronym] = meaning


# ---------- Speicherung als Datei ----------
with open("{0}\\data\\preprocessing_acronymcleanup\\acronyms.jsonl".format(root.absolute()), 'w', encoding='utf-8') as file:  # Erstellen einer Jsonl-Datei, welche ein Dictionary zu Akronymen und ihrer Langform enthält
    json_data = json.dumps(acronyms, ensure_ascii=False)
    file.write(json_data)
