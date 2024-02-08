import os
import json
import spacy
from utils_reddit import sanitize_text, is_consonant, update_dict, base_in_lexica, check_token_stem_for_bert
from utils_reddit import read_prefixes, read_suffixes, str_includes_number
import time
import pandas as pd
from transformers import AutoTokenizer
import itertools
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--month', dest='month', help='month of the input file', required=True)
parser.add_argument('-y', '--year', dest='year', help='year of the input file', required=True)
args = parser.parse_args()

def get_sentences(json_path, out_excel):
    all_sentences = []

    dbmdz_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased").vocab.keys()

    lexica = read_lexica_dereko("data/DeReKo-2014-II-MainArchive-STT.100000.freq")

    prefix_list = read_prefixes("data/prefixes.txt")
    suffix_list = read_suffixes("data/suffixes.txt")

    derivate_dict = {}

    with open(json_path, "r") as f:
        nlp = spacy.load("de_core_news_sm")
        c = 0
        for line in f:
            c += 1
            if c % 100000 == 0:
                print(f"{c} posts analyzed!")
            data = json.loads(line)
            body = data["body"].strip()
            doc = nlp(body)
            for sent in doc.sents:
                clean_sent = str(sent).strip()
                clean_sent = sanitize_text(clean_sent)

                if len(sent) < 10 or len(sent) > 100:  # DagoBERT uses 10 words as minimum - applies to German??
                    continue
                else:
                    i = 0
                    for token in sent:
                        token = str(token)
                        if i == 0:
                            token = token.lower()
                        i += 1
                        if len(token) > 4 and not str_includes_number(token):
                            yes, info = check_derivate(token, lexica, dbmdz_tokenizer, prefix_list, suffix_list)
                            if yes:
                                derivate_dict = update_dict(token, clean_sent, info, derivate_dict)

    df = pd.DataFrame.from_dict(derivate_dict, orient="index")
    df.to_excel(out_excel)

    return all_sentences


def check_derivate(token, lexica, tokenizer, prefix_list, suffix_list):
    """
    Checks a token if it is a derivative
    :param token: word to be checked
    :param lexica: DeReKo lexica
    :param tokenizer: LLM tokenizer
    :param prefix_list: list of prefixes
    :param suffix_list: list of suffixes
    :return: tuple with (boolean: is a derivative, dict: info of derivative). The info for example includes {"affix": ung, "base": liefern,
                          "count": 1, "in_lexica": True, "stem": liefer, "mode": "suffix"}
    """

    words_without_match = dict()
    words_with_more_matches = dict()

    prefixes, suffixes = get_affixes_for_word(token, prefix_list, suffix_list)
    if len(prefixes) == 0 and len(suffixes) == 0:  # words without affix
        words_without_match[token] = []

    elif len(prefixes) == 1 and len(suffixes) == 0:  # words with exactly one prefix
        prefix = prefixes[0]
        res = analyze_prefix(token, prefix, lexica, tokenizer)
        if res:
            return res

    elif len(prefixes) == 0 and len(suffixes) == 1:  # words with exactly one suffix
        suffix = suffixes[0]
        res = analyze_suffix(token, suffix, lexica, tokenizer)
        if res:
            return res


    elif len(prefixes) == 1 and len(suffixes) == 1:  # word with exactly one prefix and suffix
        prefix = prefixes[0]
        suffix = suffixes[0]
        res = analyze_prefix_suffix(token, prefix, suffix, lexica, tokenizer)
        if res:
            return res


    elif len(prefixes) > 1 and len(suffixes) == 0:  #words with multiple prefixes
        prefixes.sort(key=len, reverse=True)
        results = []

        for prefix in prefixes:  # check each prefix
            res = analyze_prefix(token, prefix, lexica, tokenizer)
            if res:
                if res[1]["in_lexica"]:
                    # print(f"Stem in lexica! {res[1]['affix']} {res[1]['stem']}")
                    return res
                else:
                    results.append(res)

        if len(results) > 0:
            if len(results) > 1:
                if results[0][1]["affix"] == "erz" and results[1][1]["affix"] == "er":
                    return results[1]
            return results[0]

    elif len(prefixes) == 0 and len(suffixes) > 1:  #words with multiple suffixes
        suffixes.sort(key=len)
        results = []

        for suffix in suffixes:  # check each suffix
            res = analyze_suffix(token, suffix, lexica, tokenizer)
            if res:
                if res[1]["in_lexica"]:
                    return res
                else:
                    results.append(res)

        if len(results) > 0:
            if "ier" in suffixes and "er" in suffixes and len(results) > 1:
                return results[1]
            return results[0]

    elif len(prefixes) > 1 or len(suffixes) > 1:  # words with multiple prefixes and suffixes
        results = []
        prefix_suffix_combis = list(itertools.product(prefixes, suffixes))
        for prefix, suffix in prefix_suffix_combis:
            res = analyze_prefix_suffix(token, prefix, suffix, lexica, tokenizer)
            if res:
                if res[1]["in_lexica"]:
                    #print(f"Stem in lexica! {res[1]['stem']} {res[1]['affix']}")
                    return res
                else:
                    results.append(res)

        if len(results) > 0:
            return results[0]

    else:
        words_with_more_matches[token] = (prefixes.copy(), suffixes.copy())

    return False, None # no affix found


def analyze_prefix(token, prefix, lexica, tokenizer):
    """
    Analyzes prefix with rules
    :param token: potential derivative
    :param prefix: prefix that matches token
    :param lexica: dereko lexica
    :param tokenizer: LLM tokenizer
    :return: (boolean: is prefixed derivative, dict: info)
    """
    stem = token[len(prefix):]  # get token without prefix: gemacht --> macht
    r_stem = None
    r_base = None

    if len(stem) > 3:  # otherwise re -- if
        in_lexica, good_pos, base = base_in_lexica(stem, token, lexica, tokenizer)

        if not good_pos:  # exclude grammatical words
            return

        if not prefix_elimination(stem, prefix, token, lexica):  # apply rule-based elimination
            return

        # check if stem in lexica and get base/infinitive form

        if in_lexica and check_token_stem_for_bert(token, base, tokenizer):
            r_base = base
        else:
            found_base, base = get_prefix_base_by_rules(stem, prefix, token)
            if found_base and check_token_stem_for_bert(token, base, tokenizer):
                if len(base) > 3:
                    r_base = base

        if check_token_stem_for_bert(token, stem, tokenizer):  # check if in LLM vocab
            r_stem = stem

        if r_stem or r_base:
            return True, {"affix": prefix, "base": r_base,
                          "count": 1, "in_lexica": in_lexica, "stem": r_stem, "mode": "prefix"}


def analyze_prefix_suffix(token, prefix, suffix, lexica, tokenizer):
    """
    Analyzes potential double-affixed derivative with rules
    :param token: potential derivative
    :param prefix: prefix that matches token
    :param suffix: suffix that matches token
    :param lexica: DeReKo lexica
    :param tokenizer: LLM tokenizer
    :return: (boolean: is suffixed derivative, dict: info)
    """
    stem = token[:-len(suffix)]
    stem = stem[len(prefix):]
    r_stem = None
    r_base = None

    if len(stem) > 3:  # otherwise re -- if
        in_lexica, good_pos, base = base_in_lexica(stem, token, lexica, tokenizer)
        if not good_pos:
            return

        if not (suffix_elimination(token, stem, suffix, lexica) and prefix_elimination(stem, prefix, token, lexica)):
            return

        # check if stem in lexica and get base/infinitive form
        if in_lexica and not suffix in ["end"] and check_token_stem_for_bert(token, base, tokenizer):
            r_base = base
        else:
            not_confix, base = get_suffix_base_by_rules(stem, suffix, lexica)
            if not_confix and check_token_stem_for_bert(token, base, tokenizer):
                r_base = base

        # check if stem in BERT and token not in BERT
        if check_token_stem_for_bert(token, stem, tokenizer):
            r_stem = stem

        if r_base or r_stem:
            return True, {"affix": (prefix, suffix), "base": r_base,
                              "count": 1, "in_lexica": in_lexica, "stem": r_stem, "mode": "both"}

def analyze_suffix(token, suffix, lexica, tokenizer):
    """
    Analyzes prefix with rules
    :param token: potential derivative
    :param suffix: suffix that matches token
    :param lexica: dereko lexica
    :param tokenizer: LLM tokenizer
    :return: (boolean: is suffixed derivative, dict: info)
    """
    stem = token[:-len(suffix)]
    r_stem = None
    r_base = None

    if len(stem) > 3:
        in_lexica, good_pos, base = base_in_lexica(stem, token, lexica, tokenizer)
        if not good_pos:  # rule out grammatical words
            return

        if not suffix_elimination(token, stem, suffix, lexica):  # apply suffix elimination rules
            return

        # get base
        if in_lexica and not suffix in ["end"] and check_token_stem_for_bert(token, base, tokenizer):
            r_base = base
        else:  # stem not in lexica, try rule based approach
            not_confix, base = get_suffix_base_by_rules(stem, suffix, lexica)
            if not_confix and check_token_stem_for_bert(token, base, tokenizer):
                if len(base) > 2:
                    r_base = base

        if check_token_stem_for_bert(token, stem, tokenizer):  # check if in LLM vocab
            r_stem = stem

        if r_base or r_stem:
            return True, {"affix": suffix, "base": r_base,
                          "count": 1, "in_lexica": in_lexica, "stem": r_stem, "mode": "suffix"}


def suffix_elimination(token, stem, suffix, lexica):
    """
    Rules that eliminate a suffix for a potential derivative
    :param token: potential derivative
    :param stem: stem of token
    :param suffix: suffix of token
    :param lexica: DeReKo lexica
    :return: Bool: False if elimination rule applies
    """
    if suffix == "chen" and (stem[-1] == "s" or not is_consonant(stem[-1])):  # -chen often confused with -schen
        return False

    if suffix in ["er", "ast", "at", "i", "ie", "ier", "ik", "ine", "ist", "ling", "ner", "or", "ade", "ler", "ur"] \
            and token[0].islower():  ## -er only for N --> N Politiker -weg for schlichtweg, rundweg
        return False

    if suffix in ["ern", "weg", "bar", "fach", "haft", "end", "ens", "ig", "lings", "nd", "weise", "wegen"] \
            and token[0].isupper():  # -ern only for N --> V bleiern, hungern
        return False

    if suffix == "e":
        possible_adj_form = stem.lower()  # Tiefe --> tief
        if stem[0].isupper() and possible_adj_form in lexica:
            if not lexica[possible_adj_form]["pos"] in ["ADJA", "ADJD"]:
                return False
        else:
            return False


    if suffix == "age" and (stem.endswith("sl") or stem.endswith("st") or is_consonant(stem[-3:])):
        return False

    if suffix == "ens":
        return False

    if suffix == "ette":
        test_word = stem + "et"
        if test_word in lexica:
            return False

    if suffix in ["i", "ie", "ik", "in", "ine", "ade"] and not is_consonant(stem[-1]):
        return False

    if suffix == "tel" and stem[-1] == "t":
        return False

    if suffix == "er" and stem[-1] == "i":
        return False

    return True


def prefix_elimination(stem, prefix, token, lexica):
    """
    Rules that eliminate a prefix for a potential derivative
    :param token: potential derivative
    :param stem: stem of token
    :param prefix: prefix of token
    :param lexica: DeReKo lexica
    :return: Bool: False if elimination rule applies
    """
    if prefix == "a" and stem[0] == "b":
        return False

    if prefix in ["ab", "ent", "er", "be"] and token[0].isupper():
        return False
    return True


def get_prefix_base_by_rules(stem, prefix, token):
    """
    Try to recover base of prefixed word by rules
    :param stem: stem of derivative
    :param prefix: prefix of derivative
    :param token: derivative
    :return: (boolean: base retrieval possible, base)
    """
    if is_consonant(stem):
        return False, stem

    if prefix in ["hyper", "ko", "kon", "konter", "makro", "mikro", "neo", "para", "post", "prä", "re", "retro", "sub",
                  "trans", "über", "un", "unter", "wider"]:
        return False, stem

    if prefix in ["er", "ent", "ab", "be", "an", "de", "hinter", "miss", "um", "ver",
                  "zer"] and not stem.startswith("ge") and token[0].islower():
        if len(stem) > 6 and stem.endswith("ende"):
            return True, stem[:-2]
        elif len(stem) > 6 and stem.endswith("enden"):
            return True, stem[:-3]
        elif stem.endswith("en") or stem.endswith("ern"):
            return True, stem

    if prefix in ["erz", "ex", "inter", "in", "im", "il", "multi", "ultra"]:
        if stem.endswith("e"):
            return True, stem[:-1]
        elif stem.endswith("en"):
            return True, stem[:-2]
        elif stem.endswith("ion") or stem.endswith("nal"):
            return True, stem

    return False, stem

def get_suffix_base_by_rules(stem, suffix, lexica):
    """
    Try to recover base of suffixed word by rules
    :param stem: stem of derivative
    :param suffix: suffix of derivative
    :param lexica: DeReKo lexica
    :return: (boolean: base retrieval possible, base)
    """
    base = stem
    if is_consonant(stem):
        return False, stem

    # confix indicating suffix
    if suffix in ["abel", "age", "al", "ant", "anz", "ar", "ast", "at", "el", "ent", "ibel", "ie", "ik"]:
        return False, stem

    elif suffix in ["alie", "chen", "ei", "schaft", "tum", "in", "keit", "mäßig", "nis"]:
        return True, stem

    elif suffix == "ung":
        if stem.endswith("r"):
            base = stem + "n"
        else:
            base = stem + "en"  # -ung: V -> N, besetzen -> Besetzung
        return True, base.lower()

    elif suffix == "lich":
        if "äu" in stem:
            base = re.sub("ä", "a", stem)
            return True, base.title()
        elif "ö" in stem:
            base = re.sub("ö", "o", stem)
            return True, base.title()
        elif stem.endswith("er"):
            return True, stem + "n"
        else:
            return True, stem + "en"

    elif suffix == "er":
        if not stem.endswith("tik"):
            base = stem + "en"  # -er: V -> N, lachen -> Lacher
            return True, base.lower()
        elif stem.endswith("tik"):
            return True, base
        elif stem.endswith("ik"):
            base = stem[:-2]
            return True, base

        # Ausnahmen manchmal  N -> N Programmatiker, Politiker, Anhänger, Nebenkläger

    elif suffix == "ig":  # -ig: N -> ADJ, hunger -> hungrig nur wenn consonant - er
        if stem.endswith("r") and is_consonant(stem[-2]) and stem[-2] not in "rh":
            return True, stem[:-1] + "er"
        if stem + "e" in lexica:
            return True, lexica[stem + "e"]["base"]

    elif suffix == "isch" and base + "e" in lexica:
        return True, lexica[base + "e"]["base"]
    # elif suffix == "ant":  # ignorant, brisant, tolerant
    #     base = stem + "anz"

    elif suffix == "weise":
        if stem[-1] == "s":
            base = stem[:-1]
        elif stem[-2:] == "er":
            base = stem[:-2]
        return True, base

    elif suffix == "bar":
        if stem.endswith("er"):
            base = stem + "n"
        else:
            base = stem + "en"
        return True, base

    elif suffix == "e":
        return True, stem.lower()

    elif suffix in ["end", "nd"]:
        if stem.endswith("e") or stem.endswith("r") or stem.endswith("l"):
            return True, stem.lower()+"n"
        return True, stem.lower() + "en"

    elif suffix == "ur":
        if stem.endswith("e"):
            return True, stem[:-1]
        else:
            return True, stem

    elif suffix == "dings":
        if stem.endswith("er"):
            return True, stem[:-2]
        else:
            return True, stem

    elif suffix == "ell":
        if stem.endswith("in"):
            base = stem + "al"
        return True, base.title()

    elif suffix == "los":
        if stem.endswith("s"):
            base = stem[:-1]
        return True, base

    elif suffix == "ner":
        base = stem + "en"
        if base in lexica:
            return True, lexica[base]["base"]

    return False, base


def read_lexica_dereko(base_path):
    df = pd.read_csv(base_path, header=None, sep="\t")
    df.columns = ["word", "base", "pos", "frequency"]

    # remove corrupted lines, articles and punctuation
    df = df.dropna(subset=["frequency"])
    df = df.drop(df[df.base == "unknown"].index)
    df = df.drop(df[df.base == "UNKNOWN"].index)
    df = df.drop(df[df.pos.str.contains("$", regex=False)].index)

    df["word"] = df["word"].str.lower()

    s = df.set_index("word").T.to_dict("dict")
    return s


def get_affixes_for_word(word, prefix_list, suffix_list):
    """
    Checks if the word begins with prefixes or ends with suffixes
    :param word: a string
    :param prefix_list: list of all prefixes
    :param suffix_list: list of all suffixes
    :return: list with matching prefixes, list with matching suffixes
    """
    prefixes = []
    suffixes = []
    word = word.lower()  # to also match prefixed nouns
    for prefix in prefix_list:
        if word.startswith(prefix):
            prefixes.append(prefix)

    for suffix in suffix_list:
        if word.endswith(suffix):
            suffixes.append(suffix)

    return prefixes, suffixes




if __name__ == "__main__":
    start = time.time()
    month = f"0{args.month}" if int(args.month) <10 else args.month
    file_name = f"RC_{args.year}-{month}_ft"
    analyzed_path = os.path.join("german_reddit_comments", file_name+".txt")
    finished_path = os.path.join(f"derivative_lists/excel", file_name+".xlsx")
    if os.path.exists(analyzed_path):
        print(f"Start with analyzing {month}: {start}")
        get_sentences(analyzed_path, finished_path)
        end = time.time() - start
        print(f"End with analyzing after {end/60} minutes")
    else:
        print(f"Path to {file_name} not found.")

