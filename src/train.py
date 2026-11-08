# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning Qwen-2.5 sur la Darija marocaine.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models/qwen-darija-checkpoints")
    parser.add_argument("--adapter_dir", type=str, default="./models/qwen-darija-lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Chargement du tokenizer pour {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Chargement et préparation des datasets...")
    dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    def tokenize_function(batch):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in batch["messages"]]
        return tokenizer(texts, truncation=True, max_length=256)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    print("Configuration du modèle quantifié (QLoRA)...")
    use_cuda = torch.cuda.is_available()
    bnb_config = None
    device_map = None
    
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        device_map = "auto"
        torch_dtype = torch.float16
    else:
        print("⚠️ Aucun GPU détecté, exécution sur CPU sans quantification (mode test/développement).")
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    if use_cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4 if use_cuda else 1,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=use_cuda,
        use_cpu=not use_cuda,
        gradient_checkpointing=use_cuda,
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Lancement de l'entraînement...")
    trainer.train()

    print(f"Sauvegarde des adaptateurs LoRA finaux dans : {args.adapter_dir}")
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print("Entraînement et sauvegarde terminés avec succès !")

if __name__ == "__main__":
    main()