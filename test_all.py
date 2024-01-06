import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Read_Dataset import read_all, read_Q, read_H, read_A
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Llama-2-chat",device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Llama-2-chat",use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

lst = read_all()
acc = 0
size = len(lst)

for i in lst:
  Q = read_Q(i)
  H = read_H(i)
  A = read_A(i)
  input_ids = tokenizer(['<s> Question:' + Q + 'str \n</s> <s>premise:  </s> <s>hypothesis: Security forces were on high alert after a campaign marred by violence.</s>' + H '<s>Answer: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
  generate_input = {
      "input_ids":input_ids,
      "max_new_tokens":512,
      "do_sample":True,
      "top_k":50,
      "top_p":0.95,
      "temperature":0.3,
      "repetition_penalty":1.3,
      "eos_token_id":tokenizer.eos_token_id,
      "bos_token_id":tokenizer.bos_token_id,
      "pad_token_id":tokenizer.pad_token_id
  }
  generate_ids  = model.generate(**generate_input)
  text = tokenizer.decode(generate_ids[0])
  print(text)

  if(t[0] == A[0])
    acc++

print(acc/size)
