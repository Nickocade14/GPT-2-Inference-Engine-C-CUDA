import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
sequence = "The capital of France is"

inputs = tokenizer.encode(sequence, return_tensors='pt')

outputs = model.generate(inputs, max_new_tokens=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)

