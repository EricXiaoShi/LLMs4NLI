import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Llama-2-chat",device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Llama-2-chat",use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s> Question: Please identify whether the premise entails the hypothesis. The answer should be exact "entailment" or "not_entailment"\n</s> <s>premise: Security forces were on high alert after an election campaign in which more than 1,000 people, including seven election candidates, have been killed. </s> <s>hypothesis: Security forces were on high alert after a campaign marred by violence.</s> <s>Answer: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
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
