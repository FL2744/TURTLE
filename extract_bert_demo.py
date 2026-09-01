"""Download BERT once and create TURTLE's real hidden-state demonstration.

Run from the repository root:
    ./.venv/bin/python extract_bert_demo.py

The output is a compressed NumPy archive containing the complete 13 x 768
hidden-state matrix for the word "jaguar" in two different contexts.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "google-bert/bert-base-uncased"
EXAMPLES = (
    {
        "key": "animal",
        "label": "Jaguar · animal",
        "sentence": "The jaguar rested quietly in the jungle.",
        "target_word": "jaguar",
    },
    {
        "key": "car",
        "label": "Jaguar · automobile",
        "sentence": "The Jaguar accelerated quickly out of the garage.",
        "target_word": "Jaguar",
    },
)

def extract_word_trajectory(model, tokenizer, sentence, target_word):
    words = sentence.replace(".", " .").split()
    target_word_index = next(
        index for index, word in enumerate(words)
        if word.casefold() == target_word.casefold()
    )
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt")
    word_ids = encoded.word_ids(batch_index=0)
    token_indices = [i for i, word_id in enumerate(word_ids) if word_id == target_word_index]
    if not token_indices:
        raise ValueError(f"Could not locate {target_word!r} in tokenized sentence")

    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True)

    # Average subword pieces when a target word occupies multiple tokens.
    states = torch.stack([
        layer[0, token_indices, :].mean(dim=0)
        for layer in output.hidden_states
    ]).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return states.astype(np.float32), tokens, token_indices


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    archive = {}
    metadata = {"model": MODEL_NAME, "examples": []}
    for example in EXAMPLES:
        states, tokens, token_indices = extract_word_trajectory(
            model, tokenizer, example["sentence"], example["target_word"]
        )
        archive[example["key"]] = states
        metadata["examples"].append({
            **example,
            "shape": list(states.shape),
            "tokens": tokens,
            "target_token_indices": token_indices,
        })
        print(f"{example['label']}: {states.shape}")

    output_directory = Path(__file__).with_name("data")
    output_directory.mkdir(exist_ok=True)
    np.savez_compressed(output_directory / "bert_jaguar_hidden_states.npz", **archive)
    (output_directory / "bert_jaguar_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved demonstration data in {output_directory}")


if __name__ == "__main__":
    main()
