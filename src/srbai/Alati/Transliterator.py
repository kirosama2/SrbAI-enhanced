def transliterate_cir2lat(text: str) -> str:
    """
    Pretvara tekst napisan ćirilicom u latinicu
    :param text: Tekst na ćirilici
    :return: Tekst na latinici
    """
    mappings = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e", "ж": "ž", "з": "z", "и": "i",
                "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
                "с": "s", "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
                "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E", "Ж": "Ž", "З": "Z", "И": "I",
                "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj", "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R",
                "С": "S", "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č", "Џ": "Dž", "Ш": "Š"}
    translit = ""
    for char in text:
        if char in mappings.keys():
            translit = translit + mappings[char]
        else:
            translit = translit + char
    return translit


def transliterate_lat2cir(text: str) -> str:
    """
    Pretvara tekst napisan na laticini u ćirilicu
    :par