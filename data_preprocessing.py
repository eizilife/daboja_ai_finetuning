from datasets import load_dataset

def formatting_prompts_func(examples):
    """KorQuAD 데이터셋을 LLaMA 3.2 학습 형식으로 변환"""
    eos_token = '<|end_of_text|>'
    korQuAD_prompt = """### Question:
{}

### Context:
{}

### Answer:
{}"""

    instructions = examples["question"]
    inputs = examples["context"]
    outputs = [item['text'][0] for item in examples["answers"]]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = korQuAD_prompt.format(instruction, input, output) + eos_token
        texts.append(text)

    return {"text": texts}

def load_and_preprocess_data(split="train"):
    """데이터셋 로드 및 전처리"""
    print(f"Loading KorQuAD dataset ({split} split)...")
    dataset = load_dataset("KorQuAD/squad_kor_v1", split=split)

    print(f"Total samples before preprocessing: {len(dataset)}")

    print("Preprocessing dataset...")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        desc="Formatting dataset"
    )

    print(f"Sample preprocessed data:\n{dataset[0]['text'][:500]}...")

    return dataset

if __name__ == "__main__":
    train_dataset = load_and_preprocess_data("train")
    validation_dataset = load_and_preprocess_data("validation")

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")