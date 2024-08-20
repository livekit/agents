import re


# rule based segmentation based on https://stackoverflow.com/a/31505798, works surprisingly well
def split_sentences(
    text: str, min_sentence_len: int = 20
) -> list[tuple[str, int, int]]:
    """
    the text may not contain substrings "<prd>" or "<stop>"
    """
    alphabets = r"([A-Za-z])"
    prefixes = r"(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = r"(Inc|Ltd|Jr|Sr|Co)"
    starters = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = r"[.](com|net|org|io|gov|edu|me)"
    digits = r"([0-9])"
    multiple_dots = r"\.{2,}"

    # fmt: off
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>", text)
    text = re.sub(websites,"<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    # text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    # TODO(theomonnom): need improvement for ""..." dots", check capital + next sentence should not be
    # small
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)), text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(r" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(r" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(r" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text:
        text = text.replace(".”","”.")
    if "\"" in text:
        text = text.replace(".\"","\".")
    if "!" in text:
        text = text.replace("!\"","\"!")
    if "?" in text:
        text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    # fmt: on

    splitted_sentences = text.split("<stop>")
    text = text.replace("<stop>", "")

    sentences: list[tuple[str, int, int]] = []

    buff = ""
    start_pos = 0
    end_pos = 0
    for match in splitted_sentences:
        sentence = match.strip()
        if not sentence:
            continue

        buff += " " + sentence
        end_pos += len(match)
        if len(buff) > min_sentence_len:
            sentences.append((buff[1:], start_pos, end_pos))
            start_pos = end_pos
            buff = ""

    if buff:
        sentences.append((buff[1:], start_pos, len(text) - 1))

    return sentences
