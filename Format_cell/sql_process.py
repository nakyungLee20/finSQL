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

with open("/home/broodling/finQA/Results/text2SQL_0823.json", "r") as f4:
  sqls = json.load(f4)

questions = []
for ques in datas:
  questions.append(ques['qa']['question'])

queries = []
for idx in range(0,len(sqls)):
  tokens = tokenizer.tokenize(sqls[idx])
  if len(tokens) > 9500:
      quer = "Please regenerate SQL query according to the question."
      queries.append(quer)
  else:
    queries.append(sqls[idx])

print(len(questions), len(queries))


# Make schema code
def table_schema(table):
  schema = "CREATE TABLE TableD (\n"

  # 정규식을 통해 col/val 추출 및 val 정규화
  pattern = re.compile(r"<col>\s*(.*?)\s*</col>\s*<val>\s*(.*?)\s*</val>")
  matches = pattern.findall(table)
  # number_pattern = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$")
  number_pattern = re.compile(r"^-?\d+(\.\d+)?$")

  for col, val in matches:
    col_name = col.strip().replace(" ", "_").replace("-", "_").lower()
    val_clean = val.strip().replace(",", "")

    # (음수) 예외처리
    if("( " in val_clean):
       val_clean = val_clean.split("( ")[0]
       val_clean = val_clean.rstrip(" ")

    # $ 예외처리
    if("$ " in val_clean):
       val_clean = val_clean.lstrip("$ ")

    # data type 확인
    if number_pattern.match(val_clean):
        if "." in val_clean:
            col_type = "FLOAT"
        else:
            col_type = "INT"
    else:
        col_type = "VARCHAR(255)"

    schema += f"    {col_name} {col_type},\n"
  
  schema = schema.rstrip(",\n") + "\n);" # print(schema)
  
  return schema

# pre/post text schema
def text_schema(text, pos):
  schema = f"CREATE TABLE {pos} (\n"

  pattern = re.compile(r"<col>\s*(.*?)\s*</col>\s*<val>\s*(.*?)\s*</val>")
  matches = pattern.findall(text)
  # number_pattern = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$")
  number_pattern = re.compile(r"^-?\d+(\.\d+)?$")
  
  for col, val in matches:
    col_name = col.strip().replace(" ", "_").replace("-", "_").lower()
    val_clean = val.strip().replace(",", "")

    # data type 확인
    if number_pattern.match(val_clean):
        if "." in val_clean:
            col_type = "FLOAT"
        else:
            col_type = "INT"
    else:
        col_type = "VARCHAR(255)"

    schema += f"    {col_name} {col_type},\n"
  
  schema = schema.rstrip(",\n") + "\n);" # print(schema)
  
  return schema


# prompt engineering; preprocess sql query
sys_prompt = """Given the unfiltered ungrammatically incorrect SQL query and related natural language question, generate a well-formed, syntactically correct SQL query. Ensure the query follows SQL standards, with proper capitalization of SQL keywords, correct use of operators, and appropriate structure. The query should be functional and executable by a standard SQL database system. 
You must answer in following FORMAT. You can delete explanation of fixed process.:
Question: "Question"
SQLQuery: "SQL Query to be fixed"
Fixed Query: "Grammatically correct SQL query"
"""

messages =[
  {"role": "system", "content": sys_prompt},
]

filt_queries = []
for query, ques in zip(queries[:500], tqdm(questions[:500])):
  dic = {"role": "user", "content": f"""Question: {ques}\nSQLQuery: {query}\nFixed Query: """}
  messages.append(dic)

  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids = input_ids.to(base_model.device) 
  output = base_model.generate(input_ids=input_ids,
                               max_length = 12288,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response = tokenizer.decode(output)
  res = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
  res = res.rstrip("<|eot_id|>")
  if "Fixed Query:" in res:
    res = res.split("Fixed Query:")[1]
  res = res.lstrip("\n")
  filt_queries.append(res)
  del messages[-1]


with open("filtered_sql_0926_500.json", "w") as fw:
  json.dump(filt_queries, fw)

