import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_config, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import re
import sqlite3

# Call model/tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map='auto', 
    torch_dtype=torch.bfloat16)

tokenizer.pad_token = tokenizer.eos_token


# Process generated SQL query
with open("/home/broodling/finQA/datasets/FinQA/dataset/train.json", "r") as q:
  datas = json.load(q)

questions = []
for ques in datas[:200]:
  questions.append(ques['qa']['question'])

with open("/home/broodling/finQA/pre_schemas_0928.json", "r") as f1:
  preS = json.load(f1)

with open("/home/broodling/finQA/pre_values_0928.json", "r") as f2:
  preV = json.load(f2)

with open("/home/broodling/finQA/table_schemas_0928.json", "r") as f3:
  tableS = json.load(f3)

with open("/home/broodling/finQA/table_values_0928.json", "r") as f4:
  tableV = json.load(f4)

with open("/home/broodling/finQA/post_schemas_0928.json", "r") as f5:
  postS = json.load(f5)

with open("/home/broodling/finQA/post_values_0928.json", "r") as f6:
  postV = json.load(f6)

print(len(questions), len(preS), len(tableS), len(postS))

preVs = []
for pre in preV:
   total = ""
   for id in pre:
      total = total + id +'\n'
   preVs.append(total)

tableVs = []
for table in tableV:
   total = ""
   for id in table:
      total = total + id +'\n'
   tableVs.append(total)

postVs = []
for post in postV:
   total = ""
   for id in post:
      total = total + id +'\n'
   postVs.append(total)

# print(len(preVs), len(tableVs), len(postVs))


# prompt engineering; preprocess sql query of
sys_prompt = """Given natural language question and informations represented as sql grammar, extract relevant information and numerical number that can solve the financial question. Your goal is to refine given data and print only relevant information that need to solve the question with the same sql input format. Remember to use same sql grammar format at output.

## Input ##
Question: "Question"
Schema: "Database table schema"
Values: "Inserted values of table"

## Your Output ## 
You MUST answer in SQL Query FORMAT, same as Input format:
Extracted Schema: ""
Values: ""
"""

messages =[
  {"role": "system", "content": sys_prompt},
]

filt_info = []
for idx in tqdm(range(0,200)):
  schema = preS[idx] + "\n" + tableS[idx] + "\n" + postS[idx]
  values = preVs[idx] + "\n" + tableVs[idx] + "\n" + postVs[idx]
  dic = {"role": "user", "content": f"""Question: {questions[idx]}\nSchema: {schema}\n\nValues: {values}"""}
  # print(dic['content'])
  messages.append(dic)

  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids = input_ids.to(base_model.device) 
  output = base_model.generate(input_ids=input_ids,
                               max_length = 15000,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response = tokenizer.decode(output)
  res = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
  res = res.split("<|eot_id|>")[0]
  # print(res)
  filt_info.append(res)
  del messages[-1]


with open("retrieved_info_1002_200.json", "w") as fw:
  json.dump(filt_info, fw)

