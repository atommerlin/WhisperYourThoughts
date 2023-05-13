import torch
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu111 
print("CUDA available:", torch.cuda.is_available())
print("number of GPUs:", torch.cuda.device_count())
print("GPU-name:", torch.cuda.get_device_name(0))
