import torch
print(torch.__version__)               # should say +cu121
print(torch.cuda.is_available())       # should be True
print(torch.cuda.get_device_name(0))   # should show your GPU
