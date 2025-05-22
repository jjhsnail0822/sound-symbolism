# Personas
speaker_A = "Native Korean"
speaker_B = "Native Korean"
word_meanings = {
    "가닐가닐": "벌레가 기어가는 것처럼 살갗이 자꾸 또는 매우 간지럽고 자릿한 느낌.",
    "간드작깐드작": "무엇에 기대어 있거나 붙어 있는 작은 물체가 찬찬히 가볍게 자꾸 흔들리는 모양.",
    "건성-건성": "정성을 들이지 않고 대강대강 일을 하는 모양.",
}
key_word_type = ["onomatopoeia", "ideophone"]

prompt = f"""
You are a Korean language expert.
You are given a key word and its definition.
You need to generate a single-turn conversation between two native Korean speakers that includes the key word with appropriate context.
At least one speaker should use the key word and the key word should have the format of f"<key_word>".
Refer to the key word, meaning, and the personas' attribute.

The conversation should be in the following format:
dialogues = [
    {
        Speaker_A: dialogue_A,
        Speaker_B: dialogue_B,
    },
    {
        Speaker_A: dialogue_C,
        Speaker_B: dialogue_D,
    },
    ...
]

"""

dialogues = [
    {
        "speaker_A": "아, 모기에 물렸나 봐. 팔이 <가닐가닐> 간지러워서 참을 수가 없네.",
        "speaker_B": "그러게, 요즘 모기가 많아졌어. 긁지 말고 얼음으로 문질러봐."
    },
    {
        "speaker_A": "이 창문에 걸린 장식이 바람 불 때마다 <간드작깐드작> 흔들리네.",
        "speaker_B": "그러게, 바람이 좀만 불어도 계속 움직이니까 신경 쓰인다."
    },
    {
        "speaker_A": "이번 보고서, 제대로 검토한 거 맞아? 너무 <건성-건성> 한 것 같은데.",
        "speaker_B": "미안, 다른 일도 많아서 대충 봤더니 놓친 게 많았나 봐. 다시 확인할게."
    }
]

prompt = f"""
You are a Korean language expert.
Given the single-turn conversation generated from the model, generate two kinds of Multi-Choice question (MCQ).
At least one question must have the answer as "None of the above".
Options does not include at least two words from word_meanings.

The question should ask three points:
1. Given the key word in the conversation, what onomatopoeia or ideophone can be replaced that contains the context?
2. Given the key word as blank in the conversation, what onomatopoeia or ideophone can there be in there?
3. Imagine we create an artificial word that shares the same meaning of the key_word in a different language. Given the generated words, what word could fit the most considering the meaning of the word?


The question should be in the following format:

<Question>
<Conversation>
Speaker A:

Speaker B:
</Conversation>

(Optional) key_word meaning

Question:

Options:

</Question>

<Answer>

</Answer>
"""