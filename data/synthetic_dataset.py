import random
import os

SEED = 42
random.seed(SEED)

NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma",
    "Frank", "Grace", "Helen", "Ian", "Jack", "Kate",
]

RELATIONS = ["likes", "follows", "knows", "visits"]


def generate_example(max_facts=5):
    n_facts = random.randint(2, max_facts)
    facts = []
    existing = set()

    while len(facts) < n_facts:
        subject = random.choice(NAMES)
        relation = random.choice(RELATIONS)
        obj = random.choice(NAMES)

        if subject == obj:
            continue

        triple = (subject, relation, obj)

        if triple not in existing:
            existing.add(triple)
            facts.append(triple)

    if random.random() < 0.5:
        query = random.choice(facts)
        label = 1
    else:
        while True:
            query = (
                random.choice(NAMES),
                random.choice(RELATIONS),
                random.choice(NAMES),
            )
            if query not in existing and query[0] != query[2]:
                break
        label = 0

    sentences = [
        f"{subject} {relation} {obj} ."
        for subject, relation, obj in facts
    ]

    query_sentence = f"{query[0]} {query[1]} {query[2]} ?"
    text = " ".join(sentences) + " Query " + query_sentence

    return text, label


def generate_dataset(n_samples, output_path=None, max_facts=5):
    data = [generate_example(max_facts=max_facts) for _ in range(n_samples)]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text, label in data:
                f.write(f"{label}\t{text}\n")

    return data


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING SYNTHETIC DATASET")
    print("=" * 70)

    train = generate_dataset(10000, "data/generated/train.txt")
    validation = generate_dataset(2000, "data/generated/validation.txt")
    test = generate_dataset(2000, "data/generated/test.txt")

    print(f"Training samples   : {len(train)}")
    print(f"Validation samples : {len(validation)}")
    print(f"Test samples       : {len(test)}")

    print("\nExample samples:")
    for i in range(5):
        text, label = train[i]
        print("\nText:")
        print(text)
        print("Label:", label)

    print("\nDataset generation complete.")
