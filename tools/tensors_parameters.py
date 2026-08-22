from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate all the parameters removing duplicates and stor it in /reports/output_tensors.txt
total = 0
total_bytes = 0
with open("reports/output_tensors.txt", "w") as f:
    for name, param in model.named_parameters():
        print(f"{name:<50} | shape: {tuple(param.shape)} | stride: {param.stride()} | numel:{param.numel()} | total bytes: {param.numel() * param.element_size()}" , file=f)
        total += param.numel()
        total_bytes += (param.numel() * param.element_size())
    total_mb = total_bytes/1000000
    total_gb = total_mb/1000
    print("Total Parameters: ", total, file = f)
    print("Total Bytes: ", round(total_mb, 2), "MB", "or ", round(total_gb, 3), "GB", file = f)
    
    
# Generate all the parameters removing duplicates and stor it in /reports/output_duplicate_tensors.txt
dup_total = 0
total_dup_bytes = 0

with open("reports/output_duplicate_tensors.txt", "w") as f:
    for name, param in model.named_parameters(remove_duplicate= False):
        print(f"{name:<50} | shape: {tuple(param.shape)} | stride: {param.stride()} | numel:{param.numel()} | total bytes: {param.numel() * param.element_size()}" , file=f)
        dup_total += param.numel()
        total_dup_bytes += (param.numel() * param.element_size())
    total_dup_mb = total_dup_bytes/1000000
    total_dup_gb = total_dup_mb/1000
    print("Total Parameters: ", dup_total, file = f)
    print("Total Bytes: ", round(total_dup_mb, 2), "MB", "or ", round(total_dup_gb, 3), "GB", file = f)