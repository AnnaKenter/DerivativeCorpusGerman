import re
import html


def sanitize_text(text):
    text = html.unescape(text)
    sanitized_body = text.replace('\r', '')
    sanitized_body = sanitized_body.replace('\n', ' ')
    sanitized_body = re.sub(r'https?:\/\/\S*', '', sanitized_body, flags=re.MULTILINE)
    sanitized_body = re.sub(r'v=\S*t=\S*', '', sanitized_body, flags=re.MULTILINE)
    sanitized_body = re.sub(r'\/(r|u)\/', '', sanitized_body, flags=re.MULTILINE)
    sanitized_body = re.sub(r'\S*\/\S*\.\S*', '', sanitized_body, flags=re.MULTILINE)
    #sanitized_body = re.sub(r'\(?http[^ ]+\)?', '', sanitized_body)
    return sanitized_body


def is_consonant(word):
    for letter in word:
        if letter not in "bcdfghjklmnpqrstvwxyz":
            return False
    return True

def token_in_bert(token, tokenizer):
    return token.lower() in tokenizer

def update_dict(token, context, info, derivate_dict):
    if token in derivate_dict:
        derivate_dict[token]["count"] += 1
        derivate_dict[token]["context"].append(context)
    else:
        derivate_dict[token] = info
        derivate_dict[token]["context"] = [context]
    return derivate_dict

def base_in_lexica(stem, token, lexica, tokenizer):
    """
    Checks if the stem is in the lexica and if an infinitive/base exists
    returns the base (if found)
    """

    if check_bad_pos(stem, lexica) or check_bad_pos(token, lexica):
        return False, False, stem

    if stem.lower() in lexica:
        base = lexica[stem.lower()]["base"]
        if token_in_bert(base, tokenizer):
            return True, True, base

    return False, True, stem

def check_bad_pos(word, lexica):

    bad_pos = {"ART", "PPER", "PRF", "PIS", "PIAT", "PPOSS", "PRELS", "PRELAT", "KOUS", "KON", "KOKOM"}
    if word in lexica:
        return lexica[word]["pos"] in bad_pos
    else:
        return False


def check_token_stem_for_bert(token, stem, tokenizer):
    """
    checks that the token is not in BERT and that the stem is in BERT
    """
    return (not token_in_bert(token, tokenizer)) and token_in_bert(stem, tokenizer)


def read_prefixes(path):
    s = set()
    with open(path, encoding="utf-8") as prefix_file:
        for prefix in prefix_file:
            prefix = prefix[:-2].strip()
            s.add(prefix)
    return s

def read_suffixes(path):
    s = set()
    with open(path, encoding="utf-8") as suffix_file:
        for suffix in suffix_file:
            suffix = suffix[1:].strip()
            s.add(suffix)
    return s

def str_includes_number(token):
    r = re.search("\d", token)
    if r:
        return True
    else:
        return False



