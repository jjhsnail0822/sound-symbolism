# Higher-order factors of sound symbolism
# Excluded ipa symbols are commented out
IPA_MAP = {
        "consonants":{
            "sonorants": ["l", "m", "n"], 
            "voiced_ficatives":["v","ð",  "z",], 
            "voiceless_fricatives": ["f","s","ʃ",],
            "voiced_stops": [ "b", "d",  "ɡ" ], 
            "voiceless_stops": ["p", "t", "k" ]
        },
        "vowels": {
            "front": ["i", "ej"], 
            "back": [ "ɑ", "ow" ], 
        }
    }

IPA_TO_ALPHABET = {
    "l": "l",
    "m": "m",
    "n": "n",
    "v": "v",
    "ð": "th",
    "z": "z",
    "f": "f",
    "s": "s",
    "ʃ": "sh",
    "b": "b",
    "d": "d",
    "ɡ": "g",
    "p": "p",
    "t": "t",
    "k": "k",
    "i": "ee",
    "ej": "ay",
    "ɑ": "ah",
    "ow": "oe"
}

IPA_TO_FEATURE = {
        "l": "sonorants",
        "m": "sonorants",
        "n": "sonorants",
        "v": "voiced_ficatives",
        "ð": "voiced_ficatives",
        "z": "voiced_ficatives",
        "f": "voiceless_fricatives",
        "s": "voiceless_fricatives",
        "ʃ": "voiceless_fricatives",
        "b": "voiced_stops",
        "d": "voiced_stops",
        "g": "voiced_stops",
        "p": "voiceless_stops",
        "t": "voiceless_stops",
        "k": "voiceless_stops",
        "i": "front",
        "ej": "front",
        "ɑ": "back",
        "ow": "back"
}