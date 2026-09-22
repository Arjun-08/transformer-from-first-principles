import os
import sys

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

from data.synthetic_dataset import generate_dataset
from src.tokenizer import SimpleTokenizer
from src.model import TinyTransformer
from src.metrics import calculate_metrics
from src.utils import set_seed, count_parameters


SEED = 42

TRAIN_SIZE = 10000
VAL_SIZE = 2000
TEST_SIZE = 2000

MAX_FACTS = 5
MAX_LENGTH = 40

D_MODEL = 32
NUM_HEADS = 4
D_FF = 64
NUM_LAYERS = 2
NUM_CLASSES = 2

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


class SyntheticDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]

        ids = self.tokenizer.encode(
            text,
            self.max_length
        )

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * inputs.size(0)
            )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[:, 1]

            predictions = logits.argmax(dim=-1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )

    metrics["loss"] = average_loss

    return metrics


def main():
    print("\n")
    print("=" * 70)
    print("TINY TRANSFORMER FROM SCRATCH")
    print("=" * 70)

    set_seed(SEED)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print("\n")
    print("=" * 70)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("=" * 70)

    train_data = generate_dataset(
        TRAIN_SIZE,
        max_facts=MAX_FACTS
    )

    val_data = generate_dataset(
        VAL_SIZE,
        max_facts=MAX_FACTS
    )

    test_data = generate_dataset(
        TEST_SIZE,
        max_facts=MAX_FACTS
    )

    print(f"Training samples   : {len(train_data)}")
    print(f"Validation samples : {len(val_data)}")
    print(f"Test samples       : {len(test_data)}")

    print("\nExample training samples:")

    for i in range(3):
        text, label = train_data[i]

        print("\n--------------------------")
        print(text)
        print(f"Label: {label}")

    print("\n")
    print("=" * 70)
    print("STEP 2: BUILDING TOKENIZER")
    print("=" * 70)

    tokenizer = SimpleTokenizer()

    all_texts = [
        text for text, _ in train_data
    ]

    tokenizer.build_vocab(all_texts)

    print(
        f"Final vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    example_text = train_data[0][0]

    encoded = tokenizer.encode(
        example_text,
        MAX_LENGTH
    )

    print("\nTokenization example:")
    print("Original:")
    print(example_text)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(tokenizer.decode(encoded))

    print("\n")
    print("=" * 70)
    print("STEP 3: CREATING DATALOADERS")
    print("=" * 70)

    train_dataset = SyntheticDataset(
        train_data,
        tokenizer,
        MAX_LENGTH
    )

    val_dataset = SyntheticDataset(
        val_data,
        tokenizer,
        MAX_LENGTH
    )

    test_dataset = SyntheticDataset(
        test_data,
        tokenizer,
        MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = TinyTransformer(
        vocab_size=tokenizer.vocab_size,
        max_length=MAX_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    parameter_count = count_parameters(model)

    print(
        f"\nTotal trainable parameters: "
        f"{parameter_count:,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    print("\nOptimizer: AdamW")
    print(f"Learning rate: {LEARNING_RATE}")

    best_val_f1 = -1

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    print("\n")
    print("=" * 70)
    print("STEP 4: TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = logits.argmax(
                dim=-1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch {epoch:02d} | "
                    f"Batch {batch_idx + 1:03d}/"
                    f"{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_metrics = evaluate(
            model,
            val_loader,
            DEVICE
        )

        print("\n")
        print(f"Epoch {epoch:02d} Summary")

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss        : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Precision   : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Val Recall      : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Val F1          : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Val ROC-AUC     : "
            f"{val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab": tokenizer.token_to_id,
                "config": {
                    "max_length": MAX_LENGTH,
                    "d_model": D_MODEL,
                    "num_heads": NUM_HEADS,
                    "d_ff": D_FF,
                    "num_layers": NUM_LAYERS
                }
            }

            torch.save(
                checkpoint,
                "checkpoints/best_model.pt"
            )

            print("New best model saved.")

    print("\n")
    print("=" * 70)
    print("STEP 5: FINAL TEST EVALUATION")
    print("=" * 70)

    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_metrics = evaluate(
        model,
        test_loader,
        DEVICE
    )

    print(
        f"\nTest Loss      : "
        f"{test_metrics['loss']:.4f}"
    )

    print(
        f"Test Accuracy  : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1        : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test ROC-AUC   : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
