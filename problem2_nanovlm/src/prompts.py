LABELS = ["1", "2", "3", "4", "5"]

INSTRUCTION = ("Look at the image and answer the multiple-choice question. "
               "Respond with the number of the correct option only.")


def build_prompt(question, choices):
    options = "\n".join(f"{l}. {c}" for l, c in zip(LABELS, choices))
    return (f"{INSTRUCTION}\n\nQuestion: {question.strip()}\n"
            f"Options:\n{options}\nAnswer:")


def build_answer(idx):
    return f" {LABELS[idx]}"