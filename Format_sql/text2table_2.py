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
sys_prompt = """You are responsible for generating a SQL query that first creates a schema and then inserts values into the appropriate columns of a table based on the provided text information. The goal is to generate a grammatically correct SQL query that follows these instructions precisely.
Your goal is to:
1. Extract relevant, important information: Identify numerical values including percent(%), dollars($) from the text.
2. Create Schema: Generate a CREATE TABLE statement with an appropriate table name "PreD". You should define a corresponding column with a suitable data type. Each column should logically and concisely contain text information. Assign default values where necessary, especially if the information in the text doesn't provide values for certain columns in all rows.
3. Insert Values into the Table: Generate INSERT INTO statements to populate the table with the values derived from the text. If some values are missing for a particular column, rely on the default values if defined in schema.
4. Grammatical Correctness: Ensure the entire SQL query is grammatically correct and follows standard SQL conventions. Pay attention to proper use of commas, parentheses, and quotation marks where necessary.

The answer should follow below format:
Schema: "Create Table, schema sql here"
Values: "Insert Value, value insert sql here"
"""

user_prompt = """Text: In 2022, the company's revenue was $ (8) million. In 2023, the company achieved a revenue of $ (10) million, which was a 25 (%) increase from the previous year. The profit margin was 15%.
Table: """

assistant_prompt = """Schema:
CREATE TABLE TableD (
    year INT,
    revenue_in_millions INT,
    increase_rate INT,
    profit_margin INT
);

Values:
INSERT INTO TableD (
    year, revenue_in_millions
) VALUES (
    2022, 8
);
INSERT INTO TableD (
    year, revenue_in_millions, increase_rate, profit_margin
) VALUES (
    2023, 10, 25, 15
);
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
  #print(res1)

  pre_table.append(res1)
  del messages[-1]

## save results
with open("pre_table_0927_200.json", "w") as file1:
  json.dump(pre_table, file1)

sys_prompt2 = """You are responsible for generating a SQL query that first creates a schema and then inserts values into the appropriate columns of a table based on the provided text information. The goal is to generate a grammatically correct SQL query that follows these instructions precisely.
Your goal is to:
1. Extract relevant, important information: Identify numerical values including percent(%), dollars($) from the text.
2. Create Schema: Generate a CREATE TABLE statement with an appropriate table name "PostD". You should define a corresponding column with a suitable data type. Each column should logically and concisely contain text information. Assign default values where necessary, especially if the information in the text doesn't provide values for certain columns in all rows.
3. Insert Values into the Table: Generate INSERT INTO statements to populate the table with the values derived from the text. If some values are missing for a particular column, rely on the default values if defined in schema. Use ONLY information in TEXT when insert values.
4. Grammatical Correctness: Ensure the entire SQL query is grammatically correct and follows standard SQL conventions. Pay attention to proper use of commas, parentheses, and quotation marks where necessary.

The answer should follow below format:
Schema: "Create Table, schema sql here"
Values: "Insert Value, value insert sql here"
"""

messages =[
  {"role": "system", "content": sys_prompt2},
  {"role": "user", "content": user_prompt},
  {"role": "assistant", "content": assistant_prompt},
]

for idx in tqdm(range(0, 200)):
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
  #print(res2)

  post_table.append(res2)
  del messages[-1]

## save results
with open("post_table_0927_200.json", "w") as file2:
  json.dump(post_table, file2)

