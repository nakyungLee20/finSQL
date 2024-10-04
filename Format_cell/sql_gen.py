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



# call data => pre/post table file 바꿔야함!
with open("/home/broodling/finQA/Results/post_table_0817.json", "r") as f1:
  posts = json.load(f1)

with open("/home/broodling/finQA/Results/pre_table_0817.json", "r") as f2:
  pres = json.load(f2)

with open("/home/broodling/finQA/Results/subtable_extract_0817.json", "r") as f3:
  tables = json.load(f3)

with open("/home/broodling/finQA/datasets/FinQA/dataset/train.json", "r") as q:
  datas = json.load(q)

questions = []
for ques in datas:
  questions.append(ques['qa']['question'])

print(len(posts), len(pres), len(tables), len(questions))


# make schema code
def table_schema(table):
  schema = "CREATE TABLE TableD (\n"

  # 정규식을 통해 col/val 추출 및 val 정규화
  pattern = re.compile(r"<col>\s*(.*?)\s*</col>\s*<val>\s*(.*?)\s*</val>")
  matches = pattern.findall(table)
  number_pattern = re.compile(r"^-?\d+(\.\d+)?$")

  cols = []
  for col, val in matches:
    col_name = re.sub(r"[^\w\s]", "", col.strip())
    col_clean = col_name.replace(" ", "_").replace("-", "_").lower()
    col_clean = col_clean.rstrip("_")
    col_clean = col_clean.replace("__","_")
    val_clean = val.strip().replace(",", "")

    # null column 방지
    if len(col_clean)==0:
       continue

    # digit start 방지
    if col_clean[0].isdigit():
      stop = 0
      for i in range(1,len(col_clean)):
        if col_clean[i].isdigit():
          stop = i
        else:
          break
      col_clean = col_clean[stop+1:] + "_" + col_clean[:stop+1]
    else:
       col_clean = col_clean
    col_clean = col_clean.lstrip("_")

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

    # 중복 column 처리
    if col_clean in cols:
      col_change = col_clean +"_" + str(val_clean)
      schema += f"    {col_change} {col_type},\n"
    else:
      schema += f"    {col_clean} {col_type},\n"
    cols.append(col_clean)
  
  schema = schema.rstrip(",\n") + "\n);" # print(schema)
  
  return schema

#table_schema(example)

# pre/post text schema
def text_schema(text, pos):
  schema = f"CREATE TABLE {pos} (\n"

  pattern = re.compile(r"<col>\s*(.*?)\s*</col>\s*<val>\s*(.*?)\s*</val>")
  matches = pattern.findall(text)
  number_pattern = re.compile(r"^-?\d+(\.\d+)?$")
  
  cols = []
  for col, val in matches:
    col_name = re.sub(r"[^\w\s]", "", col.strip())
    col_clean = col_name.replace(" ", "_").replace("-", "_").lower()
    col_clean = col_clean.rstrip("_")
    col_clean = col_clean.replace("__","_")
    val_clean = val.strip().replace(",", "")
    
    # null column 방지
    if len(col_clean)==0:
       continue

    # digit start 방지
    if col_clean[0].isdigit():
      stop = 0
      for i in range(1,len(col_clean)):
        if col_clean[i].isdigit():
          stop = i
        else:
          break
      col_clean = col_clean[stop+1:] + "_" + col_clean[:stop+1]
    else:
       col_clean = col_clean
    col_clean = col_clean.lstrip("_")

    # data type 확인
    if number_pattern.match(val_clean):
        if "." in val_clean:
            col_type = "FLOAT"
        else:
            col_type = "INT"
    else:
        col_type = "VARCHAR(255)"

    # 중복 column 처리
    if col_clean in cols:
      col_change = col_clean +"_" + str(val_clean)
      schema += f"    {col_change} {col_type},\n"
    else:
      schema += f"    {col_clean} {col_type},\n"
    cols.append(col_clean)
  
  schema = schema.rstrip(",\n") + "\n);" # print(schema)
  
  return schema


# prompt engineering
sys_prompt = """Given the following pair, SQL schema tables and single natual language question, generate a correct SQL query to correctly answer the given question. Answer in ONLY SQL query format.
Use the following format:
Question: "Question"
SQLQuery: "SQL Query to run"
"""

template = """Given below SQL schemas and a natual language question, generate a corresponding SQL query.
{schema}

Question: {ques}
SQLQuery: """

messages =[
  {"role": "system", "content": sys_prompt},
]


# Run text2SQL generation (Generate SQL Query) => posts, pres, tables, questions
total_len = len(tables)
SQLs = []
schemas = []

for idx in tqdm(range(17,200)):
  table_db = table_schema(tables[idx])
  pre_db = text_schema(pres[idx], "PreD")
  post_db = text_schema(posts[idx], "PostD")

  total_db = pre_db + "\n" + table_db + "\n" + post_db
  schemas.append(total_db)
  user_prompt = template.format(schema=total_db, ques=questions[idx])
  dic = {"role": "user", "content": user_prompt}
  messages.append(dic)
  # print(dic["content"])

  input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
  input_ids = input_ids.to(base_model.device) 

  output = base_model.generate(input_ids=input_ids,
                               max_length = 12288,
                               temperature=0.2,
                               pad_token_id = tokenizer.eos_token_id)[0]
  
  response = tokenizer.decode(output)
  res = response.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
  res = res.rstrip("<|eot_id|>")
  res = res.lstrip("\n")
  # print(res)

  SQLs.append(res)
  del messages[-1]


## save result
with open("text2SQL_0926_200.json", "w") as sql:
  json.dump(SQLs, sql)

with open("schemas_0926_200.json", "w") as sch:
  json.dump(schemas, sch)

