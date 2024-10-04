import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_config, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd


## Call basemodel(llama3.1 8B) 
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map='auto', 
    torch_dtype=torch.bfloat16)

tokenizer.pad_token = tokenizer.eos_token
# print(base_model.hf_device_map)


## load data
data_pth = "/home/broodling/finQA/datasets/FinQA/dataset/train.json"

questions = []
pretext = []
posttext = []

def sequence(text_list):
  st = ""
  for line in text_list:
    if(line == "."):
      pass
    else:
      st += line + " "
  return st


with open(data_pth, "r") as file:
  data = json.load(file)

for i in data:
  questions.append(i['qa']['question'])
  pretext.append(sequence(i['pre_text']))
  posttext.append(sequence(i['post_text']))

print("Train FinQA Table Datasets: ", len(pretext), len(posttext), len(questions))
# print(pretext[0])
# print(posttext[0])


## prompt for text to table
sys_prompt = """Your job is to make a table from an unstructured text data. Extract numerical information from the provided text and generate a table based on the context and meaning of the numbers. The table should be labeled with appropriate column and row names that accurately represent the extracted data.

The input will be a text passage. Your goal is to:
1. Extract all the numbers appear in the given text. The given text might contain percentage (%), dollors ($) and other forms of number. 
2. For each number, provide a brief explanation of what the number represents or refers to in the context of the text. 
3. Based on the numbers and their explanations, define appropriate column header for a table that could represent the key information in the text. The numbers should be included in val, not in column header.
4. The column names have to be UNIQUE, which means column name should describe the number concisely(not too long) and accurately but must not be redundent!

The output SHOULD FOLLOW below format(each line indicates one extracted numbers):
|<cell>| <col> column header </col> <val> extracted number </val> |</cell>|
"""

user_prompt = """Text: In 2022, the company's revenue was $ (8) million. In 2023, the company achieved a revenue of $ (10) million, which was a 25 (%) increase from the previous year. The profit margin was 15%.
Table: """

assistant_prompt = """|<cell>| <col> Revenue (in millions) Year 2022 </col> <val> 8 </val> |</cell>|
|<cell>| <col> Revenue (in millions) Year 2023 </col> <val> 10 </val> |</cell>|
|<cell>| <col> Growth (%) Year 2023 </col> <val> 25 </val> |</cell>|
|<cell>| <col> Profit Margin (%) Year 2023 </col> <val> 15 </val> |</cell>|
"""

messages =[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": user_prompt},
  {"role": "assistant", "content": assistant_prompt},
]


# generate table's rows and columns
pre_table = []
post_table = []

for idx in tqdm(range(0, 200)):
  # pretext2table
  dic1 = {"role": "user", "content": "Text: {} \nTable: ".format(pretext[idx])}
  messages.append(dic1)
  input_ids1 = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids1 = input_ids1.to(base_model.device) 

  output1 = base_model.generate(input_ids=input_ids1,
                               max_length = 12000,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response1 = tokenizer.decode(output1)
  res1 = response1.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[2]
  res1 = res1.lstrip("\n")
  res1 = res1.rstrip("<|eot_id|>")
  print(res1)

  pre_table.append(res1)
  del messages[-1]

  # posttext2table
  dic2 = {"role": "user", "content": "Text: {} \nTable: ".format(posttext[idx])}
  messages.append(dic2)
  input_ids2 = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids2 = input_ids2.to(base_model.device) 

  output2 = base_model.generate(input_ids=input_ids2,
                               max_length = 12000,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response2 = tokenizer.decode(output2)
  res2 = response2.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[2]
  res2 = res2.lstrip("\n")
  res2 = res2.rstrip("<|eot_id|>")
  print(res2)

  post_table.append(res2)
  del messages[-1]


## save results
with open("pre_table_0927_test.json", "w") as file1:
  json.dump(pre_table, file1)

with open("post_table_0927_test.json", "w") as file2:
  json.dump(post_table, file2)

