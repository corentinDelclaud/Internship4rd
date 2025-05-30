import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test basic GPU operations
    print("\nTesting basic GPU operations...")
    x = torch.tensor([1, 2, 3]).cuda()
    y = torch.tensor([4, 5, 6]).cuda()
    z = x + y
    print(f"GPU tensor operation result: {z}")
    print(f"Result device: {z.device}")
else:
    print("No GPU available")
