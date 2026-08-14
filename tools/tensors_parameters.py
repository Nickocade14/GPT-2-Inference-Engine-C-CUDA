from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')

total = 0;

with open("reports/output_tensors.txt", "w") as f:
    for name, param in model.named_parameters():
        print(f"{name:<50} | shape: {tuple(param.shape)} | stride: {param.stride()} | numel:{param.numel()}", file=f)
        total += param.numel()
    print("Total parameters: ", total, file = f)