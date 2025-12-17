import torch
from load_cnn import load_cnn

DEVICE = "cpu"

cnn = load_cnn(device=DEVICE)

# Save FULL model (not state_dict)
torch.save(cnn, "cnn_encoder_full.pt")

print("✅ Full CNN model saved as cnn_encoder_full.pt")
