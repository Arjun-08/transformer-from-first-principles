import os
import sys

import torch

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from data.synthetic_dataset import generate_dataset
from src.tokenizer import SimpleTokenizer
from src.model import TinyTransformer


DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

MAX_LENGTH = 40
D_MODEL = 32
NUM_HEADS = 4
D_FF = 64
NUM_LAYERS = 2


def main():
    print("=" * 70)
    print("MODEL INSPECTION")
    print("=" * 70)

    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location=DEVICE
    )

    vocabulary = checkpoint["vocab"]

    tokenizer = SimpleTokenizer()
    tokenizer.token_to_id = vocabulary
    tokenizer.id_to_token = {
        value: key
        for key, value in vocabulary.items()
    }

    model = TinyTransformer(
        vocab_size=len(vocabulary),
        max_length=MAX_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        num_classes=2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    test_data = generate_dataset(
        10,
        max_facts=5
    )

    print("\n")
    print("=" * 70)
    print("INDIVIDUAL PREDICTIONS")
    print("=" * 70)

    with torch.no_grad():
        for text, label in test_data:
            ids = tokenizer.encode(
                text,
                MAX_LENGTH
            )

            input_tensor = torch.tensor(
                [ids],
                dtype=torch.long
            ).to(DEVICE)

            logits = model(input_tensor)

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            prediction = (
                probabilities
                .argmax(dim=-1)
                .item()
            )

            confidence = (
                probabilities[
                    0,
                    prediction
                ].item()
            )

            print("\n")
            print("-" * 70)
            print("Text:")
            print(text)

            print(f"\nActual label     : {label}")
            print(f"Predicted label  : {prediction}")
            print(
                f"Confidence       : "
                f"{confidence:.4f}"
            )

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
