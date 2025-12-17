import torch
from PIL import Image
import torchvision.transforms as T

from load_cnn import load_cnn
from load_llm import load_llm

DEVICE = "cpu"

cnn = load_cnn(DEVICE)
llm, tokenizer = load_llm(DEVICE)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def generate_report(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _ = cnn(image)  # features extracted (not text)

    prompt = (
        "You are an experienced radiologist.\n"
        "Generate a concise chest X-ray report.\n\n"
        "Format strictly as:\n"
        "FINDINGS:\n"
        "- ...\n\n"
        "IMPRESSION:\n"
        "- ..."
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    report = tokenizer.decode(output[0], skip_special_tokens=True)

    # 🔒 SAFETY: ensure prompt is never returned
    if prompt.strip() in report:
        report = report.replace(prompt.strip(), "").strip()

    return report
