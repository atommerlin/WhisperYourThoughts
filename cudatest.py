import torch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 
print("CUDA verfügbar:", torch.cuda.is_available())
print("Anzahl der GPUs:", torch.cuda.device_count())
print("GPU-Name:", torch.cuda.get_device_name(0))
