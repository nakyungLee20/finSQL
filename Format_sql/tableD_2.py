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
      cell = table[0][0] +" " + table[i][0] + " of " + table[0][j] + " is " + table[i][j] + "\n"
      sen += cell
  sen = sen.replace("  ", " ")
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
        cell = exam[0][0] +" " + exam[i][0] + " of " + exam[0][j] + " is " + exam[i][j] + "\n"
        fewshot += cell
fewshot = fewshot.replace("  ", " ")
# print(fewshot)

# prompt for table decomposition
sys_prompt = """You are responsible for generating a SQL query that first creates a schema and then inserts values into the appropriate columns of a table based on the provided text and question. The goal is to extract only relevant information from the text and to generate a grammatically correct SQL query that can be executed.
Your goal is to:
1. Extract relevant, important information from the given text and question: Considering question, extract only relative information or numerical data from the given text to answer the question. 
2. Create Schema: Generate a CREATE TABLE statement with an appropriate table name "TableD". You should define a corresponding column with a suitable data type. Each column should logically and concisely contain text information. Assign default values where necessary, especially if the information in the text doesn't provide values for certain columns in all rows.
3. Insert Values into the Table: Generate INSERT INTO statements to populate the table with the values derived from the text. If some values are missing for a particular column, rely on the default values if defined in schema. Use ONLY information in TEXT when insert values.
4. Grammatical Correctness: Ensure the entire SQL query is grammatically correct and follows standard SQL conventions. Pay attention to proper use of commas, parentheses, and quotation marks where necessary.

The answer must follow below format:
Schema: "Create Table, schema sql here"
Values: "Insert Value, value insert sql here"
"""

# provide example (few-shot 1)
user_prompt = """Question: what was the percentage cumulative total return for the five year period ended 31-dec-2017 of citi common stock?
Text: {}
Generated Query:
""".format(fewshot)

assistant_prompt = """Schema:
CREATE TABLE TableData (
    date DATE,
    stock_value DECIMAL(10, 2)
);

Values:
INSERT INTO TableData (date, stock_value) VALUES 
('2012-12-31', 100.0),
('2013-12-31', 131.8),
('2014-12-31', 137.0),
('2015-12-31', 131.4),
('2016-12-31', 152.3),
('2017-12-31', 193.5);
"""

messages =[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": user_prompt},
  {"role": "assistant", "content": assistant_prompt},
]


## Generate answers
answers = []
for idx in tqdm(range(0, 200)):
  dic = {"role": "user", "content": "Question: {} \nText: {} \nGenerated Query:".format(questions[idx], table_str[idx])}
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
  #print(res)

  answers.append(res)
  del messages[-1]


## save result
with open("table_extract_0928_200.json", "w") as f1:
  json.dump(answers, f1)

# 간접적인 영향력이 있었을 수 있는 케이스들도 추가로 뽑아서 돌려보기
# COT 추가?