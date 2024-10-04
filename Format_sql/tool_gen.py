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
with open("/home/broodling/finQA/datasets/FinQA/dataset/train.json", "r") as f1:
  datas = json.load(f1)

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

# prompt engineering => use 3 tools => tool system prompt 
sys_prompt = """You are a masterful tool user, you must assist the user with their queries, but you must also use the provided tools when necessary. You must reply in a concise and simple way and must always follow the provided rules.

===========================================================

Tool Library:

- execute_sqlite3(question) - Use this function to generate sql query and execute in sqlight3 library. You pass the question string here and background information needed to solve the question will always provided in sql grammar for every question. Based on  information, this function generate proper sql query and execute it to get the final result. Make sure your generated sql query is precise and the final answer must be number or yes/no format.

- equation_calcuation(question) - This function is specifically designed for creating equation and conducting math calculations. Based on provided information and passed question, first, find out numbers for making equations. Then, use [divide(/), add(+), subtract(-), multiply(*)] operator and build equations step by step. Lastly, call code_python function or do calculation to get final result. The final answer must be number.

- code_python(question) - Use this function when you need python code function. Write python code to solve the mathematical operation or do calculation without any numerical error. This function usually helps numerical calculation in complex equations. 

===========================================================

To use these tools you must first CHOOSE SUITABLE FUNCTION to solve the problem and output func_name(question) in middle of generation. Then, in Answer, you MUST output ONLY the answer(number, yes/no format) but NOT the specific function contents.


Example inputs:
Given Information: "sql database schema and its table values are given"
Question: "What is the average age for all students who do not own any pets?"

Example outputs1:
Function: execute_sqlite3("What is the average age for all students who do not own any pets?")
Function Content: SELECT avg(age) FROM student WHERE student_id NOT IN (SELECT T1.student_id FROM student AS T1 JOIN has_pet AS T2 ON T1.student_id = T2.student_id)
Answer: 22.5

Example outputs2:
Function: equation_calcuation("What is the average age for all students who do not own any pets?")
Function Content: The students who do not own any pets are not in the has_pet table which are (2, 'Lee', 'Hana', 22, 'F', 102, 302, 'BUS'), (4, 'Choi', 'Yuna', 23, 'F', 104, 304, 'DAE'). The ages of each student are 22, 23. The average age of those students are (22+23)/2 = 45/2 = 22.5
Answer: 22.5

===========================================================

Note that you must always follow the provided rules and output in the given manner. Using a function is not always necessary, use only when needed."""


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

user_prompt = """Given information: {sc}\n{val}
Question: {que}\n""".format(sc=sch, val=vals, que=task)

assistant_prompt = """Function: equation_calcuation({que})
Function Content: The students who do not own any pets are not in the has_pet table which are (2, 'Lee', 'Hana', 22, 'F', 102, 302, 'BUS'), (4, 'Choi', 'Yuna', 23, 'F', 104, 304, 'DAE'). The ages of each student are 22, 23. The average age of those students are (22+23)/2 = 45/2 = 22.5
Answer: 22.5
""".format(que=task)

messages =[
  {"role": "system", "content": sys_prompt},
  {"role": "user", "content": user_prompt},
  {"role": "assistant", "content": assistant_prompt},
]

answers = []
for idx in tqdm(range(0,200)):
  db_schema = preS[idx]+tableS[idx]+postS[idx]
  vals = preVs[idx] + tableVs[idx] + postVs[idx]
  user_prompt = """Given information: {sc}\n{val}\n\nQuestion: {que}\n""".format(sc=db_schema, val=vals, que=questions[idx])
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
with open("final_answer_1004_tools.json", "w") as file:
  json.dump(answers, file)
