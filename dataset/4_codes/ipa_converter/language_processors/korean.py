import re


def normalize_text(text):
    """모든 종류의 공백 문자를 일반 공백으로 정규화합니다."""
    return re.sub(r"\s+", " ", text).strip()


def process_text(text):
    try:
        from num2words import num2words
    except ImportError:
        raise ImportError(
            "num2words package is required. Please run 'pip install num2words'"
        )

    text = normalize_text(text)

    # 숫자를 한국어 단어로 변환 (앞뒤 공백 추가)
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='ko')} ", text)

    # 연속된 공백을 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Korean match
korean_match = re.compile(r"[가-힣]+")


def transliterate(sentence, epi, eng_epi=None):
    sentence = process_text(sentence)
    current_sentence = ""
    current_eng_sentence = ""
    ipa = ""
    for ch in sentence:
        # 영어 문자인지 확인
        if ch.isascii() and (
            ch.isalpha() or ch.isspace()
        ):  # 만약 알파벳이거나 space이고 한국어가 아니라면(특수기호 등)
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
