import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_config, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import re

# Call model/tokenizer
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map='auto', 
    torch_dtype=torch.bfloat16)

tokenizer.pad_token = tokenizer.eos_token
# print(base_model.hf_device_map)

# call data
with open("/home/broodling/finQA/post_schemas_0928.json", "r") as f1:
  posts = json.load(f1)

with open("/home/broodling/finQA/pre_schemas_0928.json", "r") as f2:
  pres = json.load(f2)

with open("/home/broodling/finQA/table_schemas_0928.json", "r") as f3:
  tables = json.load(f3)

with open("/home/broodling/finQA/datasets/FinQA/dataset/train.json", "r") as f4:
  datas = json.load(f4)

questions = []
for ques in datas[:200]:
  questions.append(ques['qa']['question'])

print(len(posts), len(pres), len(tables), len(questions))


# prompt engineering
sys_prompt = """Given the following pair of SQL schema and natual language question, generate a correct SQL query to correctly answer the given question. Considering given schema information, especially column names, generate SQL query that can solve given question and syntactically perfect(must be executed without any errors). 
Use the following format and ONLY answer in sql query:
Question: "Question"
SQLQuery: "SQL Query to run"
"""

template = """Given the SQL schema and a natual language question, generate a corresponding SQL query.
Schema: {schema}\n
Question: {ques}
SQLQuery: """

## few-shot example (BookSQL)
sch1="""CREATE TABLE chart_of_accounts(
    id INTEGER ,
    businessID INTEGER NOT NULL,
    Account_name TEXT NOT NULL,
    Account_type TEXT NOT NULL,
);
CREATE TABLE master_txn_table(
    id INTEGER ,
    businessID INTEGER NOT NULL ,
    Transaction_ID INTEGER NOT NULL,
    Transaction_DATE DATE NOT NULL,
    Transaction_TYPE TEXT NOT NULL,
    Amount DOUBLE NOT NULL,
    Account TEXT NOT NULL,
    Due_DATE DATE,             
);"""
user_prompt_1 = template.format(schema=sch1, ques="What acount had our biggest expense This week to date?")
assistant_prompt_1 = "SELECT account, SUM(debit) FROM master_txn_table AS T1 JOIN chart_of_accounts AS T2 ON T1.account = T2.account_name  WHERE account_type IN ('Expense','Other Expense') AND transaction_date BETWEEN date( current_date, \"weekday 0\", \"-7 days\") AND date( current_date) GROUP BY account ORDER BY SUM(debit) DESC LIMIT 1"

sch2 = """CREATE TABLE student (
    student_id INTEGER,
    last_name TEXT,
    first_name TEXT,
    age INTEGER,
    sex TEXT,
    major INTEGER,
    advisor INTEGER, 
    city_code TEXT, 
);
CREATE TABLE has_pet (
    student_id INTEGER,
    pet_id INTEGER,
);"""
user_prompt_2 = template.format(schema=sch2, ques="What is the average age for all students who do not own any pets?")
assistant_prompt_2 = "SELECT avg(age) FROM student WHERE student_id NOT IN (SELECT T1.student_id FROM student AS T1 JOIN has_pet AS T2 ON T1.student_id = T2.student_id)"

messages =[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": user_prompt_1},
  {"role": "assistant", "content": assistant_prompt_1},
  {"role": "user", "content": user_prompt_2},
  {"role": "assistant", "content": assistant_prompt_2},
]


# Run text2SQL generation (Generate SQL Query) => posts, pres, tables, questions
SQLs = []
for idx in tqdm(range(0,200)):
  total_db = pres[idx] + tables[idx] + posts[idx]
  user_prompt = template.format(schema=total_db, ques=questions[idx])
  dic = {"role": "user", "content": user_prompt}
  messages.append(dic)
  # print(dic["content"])

  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids = input_ids.to(base_model.device) 

  output = base_model.generate(input_ids=input_ids,
                               max_length = 12500,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response = tokenizer.decode(output)
  #print(response)
  res = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[3]
  res = res.rstrip("<|eot_id|>")
  res = res.lstrip("\n")
  #print(res)

  SQLs.append(res)
  del messages[-1]


# save results
with open("text2sql_0929_200.json", "w") as sql:
  json.dump(SQLs, sql)
