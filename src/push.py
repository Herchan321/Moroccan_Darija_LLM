# -*- coding: utf-8 -*-
import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def push_to_hub(repo_name, adapter_dir):
    print(f"Connexion et envoi de l'adaptateur LoRA de {adapter_dir} vers Hugging Face Hub: {repo_name}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    config = PeftConfig.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    print("Adaptateur envoyé avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, required=True, help="Exemple: nom_utilisateur/qwen-darija-lora")
    parser.add_argument("--adapter_dir", type=str, default="./models/qwen-darija-lora")
    args = parser.parse_args()
    push_to_hub(args.repo_name, args.adapter_dir)