from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def build_dataset(path: str, tokenizer, max_length: int):
    dataset = load_dataset("json", data_files=path, split="train")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", required=True, help="JSONL with a 'text' field")
    parser.add_argument("--output-dir", default="train/outputs/adapters/hf")
    parser.add_argument(
        "--quantization",
        choices=["none", "qlora"],
        default="none",
        help="Use QLoRA (4-bit) if set to qlora.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.quantization == "qlora":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("bitsandbytes is required for QLoRA") from exc

        try:
            import bitsandbytes as _bnb  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "bitsandbytes is not available on this platform. "
                "Use train-hf without --quantization or run QLoRA on a CUDA GPU."
            ) from exc

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=quant_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    dataset = build_dataset(args.dataset, tokenizer, args.max_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
