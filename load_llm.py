from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_llm(device):
    model_path = "models/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    return model, tokenizer
