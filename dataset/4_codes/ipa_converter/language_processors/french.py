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

    # 숫자를 프랑스어 단어로 변환 (앞뒤 공백 추가)
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='fr')} ", text)

    # 아포스트로피 통일
    text = re.sub(r"['＇’]", "'", text)

    # 특수문자를 공백으로 변환 (로마자 알파벳과 모든 종류의 아포스트로피 보존)
    text = re.sub(r"[^a-zA-ZÀ-ÿĀ-ſƀ-ƿǀ-ɏ'\" ]+", " ", text)

    # 연속된 공백을 하나로 통일
    text = re.sub(r"\s+", " ", text).strip()

    return text


def transliterate(sentence, epi, eng_epi=None):
    sentence = process_text(sentence)
    return epi.transliterate(sentence)
