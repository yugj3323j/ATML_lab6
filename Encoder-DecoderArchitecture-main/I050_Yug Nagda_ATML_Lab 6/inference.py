from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def translate_to_hindi(text: str, model, tok: dict) -> str:
    eng_seq = tok["eng_tokenizer"].texts_to_sequences([text.lower()])
    eng_pad = pad_sequences(eng_seq, maxlen=tok["max_eng_len"], padding="post")

    hin_word_index = tok["hin_tokenizer"].word_index
    hin_index_word = {v: k for k, v in hin_word_index.items()}

    # Feed full-length decoder input (start token at 0, rest zeros)
    decoder_input = np.zeros((1, tok["max_hin_len"]))
    decoder_input[0, 0] = hin_word_index.get("start", hin_word_index.get("<start>", 1))

    pred = model.predict([eng_pad, decoder_input], verbose=0)
    tokens = np.argmax(pred[0], axis=-1)

    stop_words = {"start", "end", "<start>", "<end>"}
    words = []
    for t in tokens:
        if t == 0:
            continue
        word = hin_index_word.get(t, "")
        if not word or word in stop_words:
            continue
        words.append(word)

    return " ".join(words) if words else "Translation unavailable"


def translate_to_spanish(text: str, tokenizer, model) -> str:
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def summarize_text(text: str, model, sum_tok: dict) -> str:
    tokenizer = (
        sum_tok.get("tokenizer")
        or sum_tok.get("x_tokenizer")
        or sum_tok.get("src_tokenizer")
        or list(sum_tok.values())[0]
    )
    max_text_len = (
        sum_tok.get("max_text_len")
        or sum_tok.get("max_len")
        or sum_tok.get("max_input_len")
        or 80
    )

    # Get summary tokenizer and max summary len
    sum_tokenizer = (
        sum_tok.get("summary_tokenizer")
        or sum_tok.get("y_tokenizer")
        or sum_tok.get("tgt_tokenizer")
        or tokenizer
    )
    max_sum_len = (
        sum_tok.get("max_summary_len")
        or sum_tok.get("max_output_len")
        or 20
    )

    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=max_text_len, padding="post")

    sum_word_index = sum_tokenizer.word_index
    sum_index_word = {v: k for k, v in sum_word_index.items()}

    # Decoder seed
    target_seq = np.zeros((1, max_sum_len))
    start_token = sum_word_index.get(
        "sostok", sum_word_index.get("start", sum_word_index.get("<start>", 1))
    )
    target_seq[0, 0] = start_token

    pred = model.predict([padded, target_seq], verbose=0)
    tokens = np.argmax(pred[0], axis=-1)

    stop_token = sum_word_index.get(
        "eostok", sum_word_index.get("end", sum_word_index.get("<end>", 0))
    )
    words = []
    for t in tokens:
        if t == stop_token or t == 0:
            break
        word = sum_index_word.get(t, "")
        if word and word not in ("sostok", "eostok", "start", "end", "<start>", "<end>"):
            words.append(word)

    result = " ".join(words)
    return result.strip() if result.strip() else "Summary unavailable"

