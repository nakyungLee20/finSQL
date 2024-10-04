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
data_pth = "/home/broodling/finQA/datasets/FinQA/dataset/test.json"

tables_ori = []
questions = []

with open(data_pth, "r") as file:
  data = json.load(file)

for i in data:
  questions.append(i['qa']['question'])
  tables_ori.append(i['table'])

print("Train FinQA Table Datasets: ", len(tables_ori))
# print(tables_ori[1])


## preprocess data
table_str = []
for table in tables_ori:
  sen = ""
  for i in range(1, len(table)):
    for j in range(1, len(table[0])):
      col = " <col> " + table[0][0] +" " + table[i][0] + " " + table[0][j] +  " </col>"
      val = " <val> " + table[i][j] + " </val> "
      sen += "<|cell|>" + col + val + "<|/cell|>\n"
  #print(sen)
  table_str.append(sen)

print("Table into string format: ", len(table_str))


## prompt for table decomposition
# few-shot example
exam = [[
    "date",
    "citi",
    "s&p 500",
    "s&p financials"
],
[
    "31-dec-2012",
    "100.0",
    "100.0",
    "100.0"
],
[
    "31-dec-2013",
    "131.8",
    "132.4",
    "135.6"
],
[
    "31-dec-2014",
    "137.0",
    "150.5",
    "156.2"
],
[
    "31-dec-2015",
    "131.4",
    "152.6",
    "153.9"
],
[
    "31-dec-2016",
    "152.3",
    "170.8",
    "188.9"
],
[
    "31-dec-2017",
    "193.5",
    "208.1",
    "230.9"
]]

fewshot = ""
for i in range(1, len(exam)):
    for j in range(1, len(exam[0])):
        col = " <col> " + exam[0][0] +" " + exam[i][0] + " " + exam[0][j] +  " </col>"
        val = " <val> " + exam[i][j] + " </val> "
        fewshot += "<|cell|>" + col + val + "<|/cell|>\n"
# print(fewshot)

# prompt for table decomposition
sys_prompt = """You will receive a text along with a table/query pair. When you received this pair, you should extract necessary information, especially about NUMBERS from the table, which is needed to solve the query. Then, make a condensed(summerized) table with the extracted information into a following table format. In table format, each cell starts with <|cell|> token and ends with <|/cell|> tokens. Cell's column information are enclosed by <col>, </col> tokens and value(or number) is enclosed by <val>, </val> tokens. DO NOT provide the equation to solve the problem, just correctly select the relevant columns/values. Skip detailed information and ONLY answer the extracted information with the table format."""

# provide example (few-shot 1)
user_prompt = """Query: what was the percentage cumulative total return for the five year period ended 31-dec-2017 of citi common stock?
Table: {}
Sub-Table: 
""".format(fewshot)

assistant_prompt = """|<cell>| <col> date 31-dec-2012 citi </col> <val> 100.0 </val> |</cell>|\n |<cell>| <col> date 31-dec-2012 s&p 500 </col> <val> 100.0 </val> |</cell>|\n <|cell|> <col> date 31-dec-2012 s&p financials </col> <val> 100.0 </val> <|/cell|>\n <|cell|> <col> date 31-dec-2017 citi </col> <val> 193.5 </val> <|/cell|>\n <|cell|> <col> date 31-dec-2017 s&p 500 </col> <val> 208.1 </val> <|/cell|>\n <|cell|> <col> date 31-dec-2017 s&p financials </col> <val> 230.9 </val> <|/cell|>\n """

messages =[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": user_prompt},
  {"role": "assistant", "content": assistant_prompt},
]


## Generate answers
answers = []
for idx in tqdm(range(0, len(table_str))):
  dic = {"role": "user", "content": "Query: {} \nTable: {} \nSub-Table: ".format(questions[idx], table_str[idx])}
  messages.append(dic)
  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids = input_ids.to(base_model.device) 

  output = base_model.generate(input_ids=input_ids,
                               max_length = 12000,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response = tokenizer.decode(output)
  res = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[2]
  res = res.lstrip("\n")
  res = res.rstrip("<|eot_id|>")
  # print(res)

  answers.append(res)
  del messages[-1]


## save result
with open("subtable_extract_0902_test.json", "w") as f1:
  json.dump(answers, f1)

# 간접적인 영향력이 있었을 수 있는 케이스들도 추가로 뽑아서 돌려보기
# COT 추가?