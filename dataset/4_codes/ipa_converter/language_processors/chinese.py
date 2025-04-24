import re
from opencc import OpenCC


import itertools


def normalize_text(text):
    """모든 종류의 공백 문자를 일반 공백으로 정규화합니다."""
    return re.sub(r"\s+", " ", text).strip()


def num2chinese(num, big=False, simp=True, o=False, twoalt=False):
    """
    Converts numbers to Chinese representations.
    `big`   : use financial characters.
    `simp`  : use simplified characters instead of traditional characters.
    `o`     : use 〇 for zero.
    `twoalt`: use 两/兩 for two when appropriate.
    Note that `o` and `twoalt` is ignored when `big` is used,
    and `twoalt` is ignored when `o` is used for formal representations.
    """
    # check num first
    nd = str(num)
    if abs(float(nd)) >= 1e48:
        raise ValueError("number out of range")
    elif "e" in nd:
        raise ValueError("scientific notation is not supported")
    c_symbol = "正负点" if simp else "正負點"
    if o:  # formal
        twoalt = False
    if big:
        c_basic = "零壹贰叁肆伍陆柒捌玖" if simp else "零壹貳參肆伍陸柒捌玖"
        c_unit1 = "拾佰仟"
        c_twoalt = "贰" if simp else "貳"
    else:
        c_basic = "〇一二三四五六七八九" if o else "零一二三四五六七八九"
        c_unit1 = "十百千"
        if twoalt:
            c_twoalt = "两" if simp else "兩"
        else:
            c_twoalt = "二"
    c_unit2 = "万亿兆京垓秭穰沟涧正载" if simp else "萬億兆京垓秭穰溝澗正載"
    revuniq = lambda l: "".join(k for k, g in itertools.groupby(reversed(l)))
    nd = str(num)
    result = []
    if nd[0] == "+":
        result.append(c_symbol[0])
    elif nd[0] == "-":
        result.append(c_symbol[1])
    if "." in nd:
        integer, remainder = nd.lstrip("+-").split(".")
    else:
        integer, remainder = nd.lstrip("+-"), None
    if int(integer):
        splitted = [integer[max(i - 4, 0) : i] for i in range(len(integer), 0, -4)]
        intresult = []
        for nu, unit in enumerate(splitted):
            # special cases
            if int(unit) == 0:  # 0000
                intresult.append(c_basic[0])
                continue
            elif nu > 0 and int(unit) == 2:  # 0002
                intresult.append(c_twoalt + c_unit2[nu - 1])
                continue
            ulist = []
            unit = unit.zfill(4)
            for nc, ch in enumerate(reversed(unit)):
                if ch == "0":
                    if ulist:  # ???0
                        ulist.append(c_basic[0])
                elif nc == 0:
                    ulist.append(c_basic[int(ch)])
                elif nc == 1 and ch == "1" and unit[1] == "0":
                    # special case for tens
                    # edit the 'elif' if you don't like
                    # 十四, 三千零十四, 三千三百一十四
                    ulist.append(c_unit1[0])
                elif nc > 1 and ch == "2":
                    ulist.append(c_twoalt + c_unit1[nc - 1])
                else:
                    ulist.append(c_basic[int(ch)] + c_unit1[nc - 1])
            ustr = revuniq(ulist)
            if nu == 0:
                intresult.append(ustr)
            else:
                intresult.append(ustr + c_unit2[nu - 1])
        result.append(revuniq(intresult).strip(c_basic[0]))
    else:
        result.append(c_basic[0])
    if remainder:
        result.append(c_symbol[2])
        result.append("".join(c_basic[int(ch)] for ch in remainder))
    return "".join(result)


def traditional_to_simplified(text):
    cc = OpenCC("t2s")
    return cc.convert(text)


def process_text(text):
    text = normalize_text(text)

    # 번체자를 간체자로 변환
    text = traditional_to_simplified(text)

    # 숫자를 중국어 단어로 변환 (앞뒤 공백 추가)
    text = re.sub(r"\d+", lambda m: f" {num2chinese(int(m.group()))} ", text)

    # 연속된 공백을 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transliterate(sentence, epi, eng_epi=None):
    sentence = process_text(sentence)
    current_sentence = ""
    current_eng_sentence = ""
    ipa = ""
    for ch in sentence:
        # 영어 문자인지 확인
        if ch.isascii() and (
            ch.isalpha() or ch.isspace()
        ):  # 만약 알파벳이거나 space이고 중국어가 아니라면(특수기호 등)
            if (
                current_eng_sentence or ch.isalpha()
            ):  # 만약 이미 영어가 시작했거나 alphabet(영어의 시작) 이라면. 이 때 current sentence가 비어있지 않다면 영어가 시작했다는 뜻
                current_eng_sentence += ch
                if current_sentence:
                    ipa += epi.transliterate(current_sentence)
                    current_sentence = ""
            else:  # 아니라면 (영어 중간이 아닌 특수기호 거나 영어 중간이 아닌 space)
                current_sentence += ch
            continue

        if current_eng_sentence:
            ipa += eng_epi.transliterate(current_eng_sentence)
            current_eng_sentence = ""

        current_sentence += ch

    # 마지막
    if current_eng_sentence:
        ipa += eng_epi.transliterate(current_eng_sentence)
    elif current_sentence:
        ipa += epi.transliterate(current_sentence)

    return ipa
