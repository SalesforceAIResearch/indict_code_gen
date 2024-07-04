import os
import joblib
import pdb 
import json
import re 
from tqdm import tqdm
from collections import Counter 

from datasets import load_dataset

from llm import OPENAI 

def load_data(task):
    
    action_prompt_header = ''
            
    if task == 'mitre':
        data_path = 'data/mitre.json'
        action_prompt_header = "Based on the above analysis, here's the updated version of the solution: "
    
    elif task == 'instruct':
        data_path = 'data/instruct.json'
        action_prompt_header = "Based on the above analysis, here's the updated version of the code in a single code block (wrap in ```):\n"
    
    elif task == 'autocomplete': 
        data_path = 'data/autocomplete.json'
        action_prompt_header = "Based on the above analysis, here's the updated version of the code in a single code block (wrap in ```):\n"
        
    elif task == 'promptinject':
        data_path = 'data/prompt_injection.json'
        
    elif task == 'interpreter':
        data_path = 'data/interpreter.json'
        
    elif task == 'cvs': 
        data_path = 'data/cvs.json'
        action_prompt_header = "Based on the above analysis, here's the updated version of the code in a single code block (wrap in ```):\n"
                       
    data = json.load(open(data_path, 'r'))
        
    if task in ['interpreter']:
        for sample in data:
            sample['language'] = 'python'
        
    if task in ['mitre', 'frr', 'interpreter']:
        question_prompt_key = 'mutated_prompt'
    elif task == 'promptinject':
        question_prompt_key = 'user_input'
    elif task in ['cvs']:
        question_prompt_key = 'question' 
    else:
        question_prompt_key = 'test_case_prompt'
        
    return data, action_prompt_header, question_prompt_key

def get_model(model_name, model_mapping):
    if model_name in ['gpt4', 'gpt3.5']:
        model = OPENAI(
            model_name=model_mapping[model_name],
            api_key=os.environ['OPENAI_API_KEY']
        )
    else:
        model = OPENAI(
            model_name=model_mapping[model_name],
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    return model 

def get_code_before(sample):
    code_before = ''
    line_text = sample.get("line_text", None)
    origin_code = sample.get("origin_code", None)
    if line_text and origin_code:
        code_before = origin_code.split(line_text)[0]
        code_before = code_before.strip('\n')
        
    if code_before.startswith(' ') or code_before.startswith('\t'): # indentation 
        code_before = 'if True:\n' + code_before
       
    return code_before 
        
def extract_content_in_code_blocks(input: str, keyword=''):
    # Using regular expression to find content between code blocks ```
    output = re.findall(r"```{}(.*?)```".format(keyword), input, re.DOTALL)
    if len(output)>0:
        return output[0]
    return input 

def extract_code(input: str):
    code = extract_content_in_code_blocks(input).strip()
    if code.startswith('python'):
        code = code[6:]
    return code 

def extract_tools(tool_selections):
    json_str = extract_content_in_code_blocks(tool_selections, 'json')
    try:
        tools = json.loads(json_str)
        return tools 
    except Exception:
        return []
    
def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def parse_action(string):
    if 'Search' in string:
        index = string.index('Search') + len('Search')
        string = string[index:]
        if '[' in string:
            start_idx = string.index('[') + 1 
            if ']' in string:
                end_idx = string.index(']')
                return 'Search', string[start_idx:end_idx]
            else:
                return 'Search', string[start_idx:]
        else:
            return 'Search', string
    return 'Search', string 
