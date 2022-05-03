import pandas as pd
import os.path
import re
import csv
import statistics
import sys
import nltk.corpus
from nltk. corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import emotion_cluster
# vytvoření objektu pro propojení s výsledky shlukové analýzy
object1 = emotion_cluster
# definované cesty souborů použitých lexikonů
vad_lexicon = "resources/vad_english.csv"
emoticon_lexicon = "resources/emoticon_lexicon.csv"
# způsob vyhodnocení sentimentu, buď na základě
# mediánu ("median") nebo průměru ("mean")
calculation = "median"

# statisticky vyvozené hodnoty
# o kolik proběhne navýšení arousal (přejaté z VADER analyzátoru)
# pokud je slovo napsané velkými písmeny
uppercase_increase = 0.733
# pokud je negováno jiným slovem
negation_switch = -0.74
# pokud se v textu objevuje "!" (max 4x)
exlamation_increase = 0.292
# pokud se v textu objeví ("?") 2x - 3x
question1_increase = 0.18
# pokud se v textu objeví ("?") více jak 3x
question2_increase = 0.96

# předběžná hodnota maximálního povoleného počtu emotikonů,
# které navýší hodnotu arousal (inspirováno VADER analyzátorem)
emoticon_max = 4

# funkce převede hodnotu z jedné škály na hodnotu škály druhé
# :param old_value: převáděná hodnota
# :param old_min: minimum předchozí škály
# :param old_max: maximum předchozí škály
# :param new_min: minimum nové škály
# :param new_max: maximum nové škály
def change_scale(old_value, old_min, old_max, new_min, new_max):
    return ((old_value - old_min) / (old_max - old_min)) \
           * (new_max - new_min) + new_min

# převedení výsledku nltk_postag do wordnet tvaru
# pro správnou následnou lematizaci
# :param pos: slovní druh slova vyhodnocen pomocí nltk.pos_tag
def wordnet_pos(pos):
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('N'):
        return wordnet.NOUN
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# vrátí slovní interpretaci hodnoty
# :param value: výsledná hodnota valence nebo arousal
def get_range(value):
    if value < -0.6:
        return "high negative"
    elif -0.6 <= value < -0.2:
        return "low negative"
    elif -0.2 <= value <= 0.2:
        return "neutral"
    elif 0.2 < value <= 0.6:
        return "low positive"
    elif 0.6 < value:
        return "high positive"

# hlavní funkce pro analýzu textu a emotikonů
# vrátí výsledné hodnoty valence,
# arousal a seznam vyhodnocených emocí
# :param inputText: text k analyzování
# :param mode: "cluster" (rozdělení do několika shluků)
#  nebo "single" (každá emoce je svým vlastním shlukem)
def get_sentiment (inputText, mode):
    lematizer = WordNetLemmatizer()
    error = True
    # pokud je vstup prázdný → ukončit
    if len(inputText) < 1:
        message = "Empty imput."
        return error, message

    # převedení textu na slovní tokeny
    # a vyhodnocení slovního druhu každého z nich
    words = nltk.word_tokenize(inputText)
    postag = nltk.pos_tag(words)
    # bude obsahovat hodnoty valence
    # a arousal všech prvků dokumentu obsahující sentiment
    valence_list = []
    arousal_list = []

    for count, value in enumerate(postag):
        w = value[0]
        pos = value[1]
        # lemmatizace pro každé slovo s daným slovním druhem
        lemma = lematizer.lemmatize(w, wordnet_pos(pos))

        # pokud je nějaké z 5 předchozích slov negací,
        # polarita slova bude koeficientem překlopena
        # vyjímkou jsou interpunkce mezi pozorovaným slovem a negací,
        # příp. pokud se před slovem nachází méně než 5 slov
        negation = ["no", "not", "rather", "couldn’t", "wasn’t",
                    "didn’t", "wouldn’t", "shouldn’t", "weren’t",
                    "don’t", "doesn’t", "haven’t", "hasn’t", "won’t",
                    "wont", "hadn’t", "never", "none", "nobody",
                    "nothing", "neither", "nor", "nowhere", "isn’t",
                    "can’t", "cannot", "mustn’t", "mightn’t",
                    "shan’t", "without", "needn’t"]
        neg = False
        a = 1
        b = count - 1
        while a <= 5 and b >= 0 and postag[b][0] != "." \
                and postag[b][0] != "!" and postag[b][0] != "?" \
                and postag[b][0] != ",":
            for c in negation:
                if c.casefold() == postag[b][0].casefold():
                    neg = True
            a += 1
            b -= 1

        # kontrola, zda-li slovo je napsáno velkými písmeny
        # (pouze pokud není celý text v CAPS LOCKU)
        upper = False
        if w.isupper():
            if not inputText.isupper():
                upper = True

        with open(vad_lexicon) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['word'].casefold() == lemma.casefold():
                    # pro každé slovo nalezené ve VAD lexikonu
                    # zapíše hodnoty valence a arousal
                    v = change_scale(float
                                     (row['valence']), 0, 1, -1, 1)
                    a = change_scale(float
                                     (row['arousal']), 0, 1, -1, 1)
                    # pokud negováno,
                    # překlopí polaritu dle koeficientu
                    if neg:
                        v = v * change_scale\
                            (negation_switch,-4,4,-1,1)
                    # pokud velkými písmeny,
                    # ale zbytek textu ne, přičte dodatečnou arousal
                    if upper:
                        if a > 0:
                            a = a + change_scale\
                                (uppercase_increase,-4,4,-1,1)
                        else:
                            a = a - change_scale\
                                (uppercase_increase, -4, 4, -1, 1)

                    valence_list.append(v)
                    arousal_list.append(a)
    # pokud je v textu "!" v povoleném počtu,
    # přičte dodatečnou hodnotu arousal
    exlamation_count = inputText.count("!")
    if exlamation_count > 4:
        exlamation_count = 4
    arousal_exlamation = exlamation_count * \
                         change_scale(exlamation_increase,-4,4,-1,1)
    # obdobné pro "?"
    question_count = inputText.count("?")
    arousal_question = 0
    if 1 < question_count:
        if question_count <= 3:
            arousal_question = question_count * \
                        change_scale(question1_increase,-4,4,-1,1)
        else:
            arousal_question = \
                        change_scale(question2_increase,-4,4,-1,1)

    emoticon_analyse = inputText
    emoticon_count = 0
    with open(emoticon_lexicon) as csvfile:
        reader = csv.DictReader(csvfile)
        # seřazení emotikon lexikonu podle délky emotikonů
        # (od nejdelších po nejkratší)
        reader = sorted(reader, key=lambda i: len(i['code']),
                        reverse=True)
        for row in reader:
            # hledání v textu a zapisování
            # pro každý emotikon obsažený v lexikonu
            var = row['code']
            pattern = re.compile(re.escape(var) + '+', re.IGNORECASE)
            matches = pattern.findall(emoticon_analyse)

            if len(matches) > 0:
                for match in matches:
                    emoticon_analyse = re.sub(re.escape(match), "",
                                emoticon_analyse, re.IGNORECASE)
                    dist = 1
                    v = change_scale(float(row['valence']),1,7,-1,1)
                    a = change_scale(float(row['arousal']),1,7,-1,1)

                    # zahrnutí možného prodloužení emotikonu
                    # (např. :DDD → :D:D:D)
                    if len(match) > len(var):
                        dist += len(match) - len(var)

                    for i in range (dist):
                        valence_list.append(v)
                        arousal_list.append(a)
                        emoticon_count += 1
    # přičtení dodatečného arousal pro každý další emotikon
    # do stanoveného limitu
    if emoticon_count > emoticon_max:
        emoticon_count = emoticon_max
    arousal_emoticon = emoticon_count * object1.emoticon_increase()
    # pokud v textu nenalezen žádný sentiment → ukončit
    if len(valence_list) < 1 or len(arousal_list) < 1:
        message = "No sentiment was identified from the input."
        return error, message
    # vypočtení průměru či mediánu všechn hodnot valence a arousal
    if calculation == "mean":
        valence_text = round(statistics.mean(valence_list), 3)
        arousal_text = statistics.mean(arousal_list)
    elif calculation == "median":
        valence_text = round(statistics.median(valence_list), 3)
        arousal_text = statistics.median(arousal_list)
    # přičtení arousal pro výskyt "!", "?" a vícera emotikonů
    if arousal_text > 0:
        arousal_text = round(arousal_text + arousal_exlamation
                    + arousal_question + arousal_emoticon, 3)
    else:
        arousal_text = round(arousal_text - arousal_exlamation
                    - arousal_question - arousal_emoticon, 3)

    if arousal_text > 1:
        arousal_text = 1
    if arousal_text < -1:
        arousal_text = -1
    # zjistí nejbližší shluk pro výslednou valenci
    # a arousal a vrátí vyhodnocené emoce
    cluster = object1.get_cluster\
        (valence_text,arousal_text,object1.set_cluster(mode))
    related_emotions = \
        object1.df.loc[(object1.df["cluster"] == cluster[2])]
    emotions = related_emotions["emotion"]

    return valence_text, arousal_text, emotions, related_emotions

# interakce s konzolí umožňující analýzu jednoho textu
def single_text():
    # kontroly správně zadaných vstupů
    print('\033[1m' + "Welcome to the sentiment analyzer "
        "based on VAD lexicons evaluated on Russell's "
                      "Circumplex model of affect.\n" + '\033[0m')
    input_mode = input(f'Please choose type either "' + '\033[1m' +
                       "cluster" + '\033[0m' + f'" or "' + '\033[1m'
                       + "single" + '\033[0m' + f'" mode.'
                       + '\033[1m' + "\nCluster" + '\033[0m'
                       + " mode will show a list of related "
                         "emotions to identified sentiment."
                       + '\033[1m' + "\nSingle" + '\033[0m'
                       + " mode will show the one most related "
                         "emotion to identified sentiment, "
                         "but might be less accurate: ")

    while input_mode != "single" and input_mode != "cluster":
        input_mode = input("Wrong input typed. Please type either "
                           + '\033[1m' + "cluster" + '\033[0m'
                           + " or " + '\033[1m' + "single"
                           + '\033[0m' + " mode: ")

    input_plot = input("Please choose if you also want to show "
                       "the identified emotion/s on the Circumplex"
                       " Model of Affect graph (" + '\033[1m'
                       + "yes" + '\033[0m' + "/"
                       + '\033[1m' + "no" + '\033[0m' + "): ")

    while input_plot != "yes" and input_plot != "no":
        input_plot = input("Wrong input typed. Please type either "
                           + '\033[1m' + "yes" + '\033[0m' + " or "
                           + '\033[1m' + "no" + '\033[0m'
                           + " to show the graph: ")

    inputText = input("Please " + '\033[1m' + "type your text"
                      + '\033[0m' + " to analyze its sentiment: ")
    # analyzování textů, dokud uživatel nenapíše "end"
    while inputText != "end":
        # zavolání metody pro vyhodnocení sentimentu zadaného
        # textu a vrácení výsledků v přívětivé formě
        result = get_sentiment(inputText, input_mode)
        if result[0] is True:
            print(result[1])
        else:
            print("The sentiment of the input was identified with " 
                  '\033[1m' + get_range(
                result[0]) + '\033[0m'
                  + f' valence ({result[0]}) and '
                  + '\033[1m' + get_range(
                result[1])
                  + '\033[0m' f' arousal ({result[1]}).')
            if len(result[2]) == 1:
                print("The most related emotion to this "
                      "sentiment is: " + '\033[1m', end="")
            else:
                print("The most related emotions to this "
                      "sentiment are: " + '\033[1m', end="")
            print(*result[2], sep=", ", end="")
            print('\033[0m' + ".")
            # případě zobrazí i graf
            if input_plot == "yes":
                object1.show_plot(result[3])

        inputText = input("Please " + '\033[1m'
                          + "type another text" + '\033[0m'
                          + f' to analyze its sentiment or "'
                          + '\033[1m' + "end" + '\033[0m'
                          + f'" to close the analyzer: ')

    print("Sentiment analyzer closed.")

# umožňuje provést analýzu celého souboru najednou
# obsahující vícera dokumentů
def file_analyze(input_file, mode):
    # kontrola správně zadaných vstupů, existující souboru,
    # vhodného formátu a jestli soubor není prázdný
    if len(input_file) == 0:
        sys.exit("No input file specified. Please type "
                 "the path of the file to analyze.")
    else:
        if not os.path.exists(input_file):
            sys.exit(f'The given imput file "'
                     + '\033[1m' + input_file + '\033[0m'
                     + f'" is invalid.')
        else:
            if (input_file[-4:]) != ".txt" \
                    and (input_file[-4:]) != ".csv":
                sys.exit(f'Invalid file type. Please give either "'
                         + '\x1B[3m' + "txt" + '\x1B[0m'
                         +f'" or "' + '\x1B[3m' + "csv" + '\x1B[0m'
                         + f'" file to analyze.')
            else:
                with open(input_file) as file:
                    text = file.read()
    if len(text) == 0:
        sys.exit("The file is empty.")

    if mode != "single" and mode != "cluster":
        sys.exit(f'Invalid mode. Please choose either "'
                 + '\x1B[3m' + "single" + '\x1B[0m'
                         +f'" or "' + '\x1B[3m' + "cluster"
                 + '\x1B[0m' + f'" as analyzing mode.')
    # rozdělení souboru na jednotlivé dokumenty k analyzování
    separated = text.splitlines()
    # definování názvu výstupního dokumentu, který model vytvoří
    # pokud stejný název existuje, připojí se k názvu "(1)",
    # příp. "(2)" apod.
    a = 0
    not_exist = False
    while not_exist is False:
        if os.path.exists(os.path.join(input_file.rstrip(".txtcsv")
                                + "_analyzed.csv")) and a == 0:
            a += 1
        elif os.path.exists\
                    (os.path.join(input_file.rstrip(".txtcsv")
                            + "_analyzed(" + str(a) + ").csv")):
            a += 1
        else:
            not_exist = True
    if a == 0:
        result_file = os.path.join(input_file.rstrip(".txtcsv")
                                   + "_analyzed.csv")
    else:
        result_file = os.path.join(input_file.rstrip(".txtcsv")
                                + "_analyzed(" + str(a) + ").csv")
    # vytvoření výsledného dokumentu a zapsání výsledků
    id = 1
    with open(result_file, 'w', newline= "") as file:
        header = ["id", "input", "valence","arousal",
                  "related_emotions", "commentary"]
        writer = csv.DictWriter(file, fieldnames= header)
        writer.writeheader()

        for a in separated:
            result = get_sentiment(a,mode)

            if result[0] is True:
                writer.writerow({"id": id, "input": a,
                    "valence": "n/a", "arousal": "n/a",
                    "related_emotions": "n/a",
                    "commentary": result[1]})
                id += 1
            else:
                emotions = ", ".join(result[2])
                writer.writerow({"id": id, "input": a,
                    "valence": result[0], "arousal": result[1],
                    "related_emotions": emotions,
                    "commentary": "n/a"})
                id += 1

    print(f'Sentiment analysis of given input is done. '
          f'Output file is located in "' + '\x1B[3m'
          + result_file + '\x1B[0m' + f'" csv file.')
    show_results = input("Please type " + '\033[1m' + "show"
            + '\033[0m' + " to also print the "
            "results in the console or any other "
                          "text to end the analyzer: ")
    # případně přehldně zobrazí výstupný soubor i v konzoli
    if show_results == "show":
        print(pd.read_csv(result_file).to_string())

    print("Sentiment analyzer closed.")