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

with open("/home/broodling/finQA/pre_values_0928.json", "r") as f5:
  preV = json.load(f5)

with open("/home/broodling/finQA/table_values_0928.json", "r") as f6:
  tableV = json.load(f6)

with open("/home/broodling/finQA/post_values_0928.json", "r") as f7:
  postV = json.load(f7)

questions = []
for ques in datas[:200]:
  questions.append(ques['qa']['question'])

with open("/home/broodling/finQA/text2sql_0929_200.json", "r") as f:
  sqls = json.load(f)


# prompt engineering
sys_prompt = """Given the sql table database information, sql query to execute and a question to solve, execute sql query and generate a correct answer to solve the given question. All information is represented in sql grammar format.
Your goal is to:
1. Understand structured data: Considering the given question to solve, understand given database table data, especially numerical values.
2. Execute SQL query: Execute schema creation query and value inserting queries, then execute SQL query to solve the question. 
3. Amend to final answer: Executed results or sql query may be wrong, especially when sql needs complicated mathematical calculation. You should polish up or fix the results if needed to correctly answer the question. Final answer is usually NUMBER or short format(such as yes/no). 

Use the following format and ONLY generate the final answer:
Question: "Question"
Database: "schema and inserted value information"
SQLQuery: "SQL query to run"
FinalAnswer: "Final answer based on SQL execution result"
"""

template = """Follow step by step sql reasoning. Given the database schemas, inserted values and a sql query, solve the financial question by executing SQL query. Based on executed results, correct the results if needed. The final answer is mostly number or short(yes/no) format. Do NOT print specific explanation, only give final answer.
Schema: {schema}
Values: {values}\n
Question: {ques}
SQLQuery: {query}
FinalAnswer: """

## few-shot example (BookSQL)
sch = """CREATE TABLE student (
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
vals ="""INSERT INTO student (student_id, last_name, first_name, age, sex, major, advisor, city_code) VALUES 
(1, 'Kim', 'Jin', 21, 'M', 101, 301, 'SEO'),
(2, 'Lee', 'Hana', 22, 'F', 102, 302, 'BUS'),
(3, 'Park', 'Minsoo', 20, 'M', 103, 303, 'ICN'),
(4, 'Choi', 'Yuna', 23, 'F', 104, 304, 'DAE'),
(5, 'Jung', 'Soojin', 22, 'F', 101, 305, 'GWA');

INSERT INTO has_pet (student_id, pet_id) VALUES 
(1, 501),
(3, 503),
(5, 505);
"""
task = "What is the average age for all students who do not own any pets?"
quer = "SELECT avg(age) FROM student WHERE student_id NOT IN (SELECT T1.student_id FROM student AS T1 JOIN has_pet AS T2 ON T1.student_id = T2.student_id)"
user_prompt = template.format(schema=sch, values=vals, ques=task, query=quer)
assistant_prompt = "22.5"

messages =[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": user_prompt},
  {"role": "assistant", "content": assistant_prompt},
]


answers = []
for idx in tqdm(range(0,200)):
  db_schema = pres[idx]+tables[idx]+posts[idx]
  vals = ""
  for que1 in preV[idx]:
    vals = vals + que1
  vals = vals + "\n"
  for que2 in tableV[idx]:
    vals = vals + que2
  vals = vals + "\n"
  for que3 in postV[idx]:
    vals = vals + que3
  user_prompt = template.format(schema=db_schema, values=vals, ques=questions[idx], query=sqls[idx])
  dic = {"role": "user", "content": user_prompt}
  messages.append(dic)
  # print(dic["content"])

  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids = input_ids.to(base_model.device) 

  output = base_model.generate(input_ids=input_ids,
                               max_length = 15000,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response = tokenizer.decode(output)
  # print(response)
  res = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[2]
  res = res.split("<|eot_id|>")[0]
  res = res.lstrip("\n")
  # print(res)

  answers.append(res)
  del messages[-1]


# save results
with open("final_answer_0929_200.json", "w") as file:
  json.dump(answers, file)
