import os 
import pdb 
import signal
import time

from search_engine_parser.core.engines.yahoo import Search as YahooSearch
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_experimental.utilities import PythonREPL
from llm import OPENAI 

yahoo_api = YahooSearch()
openai_api = OPENAI(
        model_name="gpt-3.5-turbo",
        api_key=os.environ['OPENAI_API_KEY']) 
wikipedia_api = DocstoreExplorer(Wikipedia())
refusal_seqs = ["i can't", "i cannot", 
        "i don't know", "i do not know", 
        "i am not sure", "i'm not sure",
        "sorry i", "i refuse"]
python_repl = PythonREPL()

class TimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    print("alarm went off")
    #return
    raise TimeoutException
    
signal.signal(signal.SIGALRM, timeout_handler)
timeout = 120  # seconds

def run_code(code):
    signal.alarm(timeout)
    try:
        result = python_repl.run(code)
        signal.alarm(0)
    except Exception as e:
        result = 'Exception: ' + str(e) 
        signal.alarm(0)
    return result 

def internet_search(query):
    search_args = (query, 1)
    try:
        results = yahoo_api.search(*search_args)
        output = {}
        for result in results:
            if 'titles' in result and result['titles']:
                output['title'] = query + ' - ' + result['titles']
            else:
                output['title'] = query 
            if 'descriptions' in result and result['descriptions']:
                output['description'] = result['descriptions']
                break 
        return output
    except Exception as e:
        print('Internet Exception:', query, '-----', e)
        return None 

def query_openai(query):
    try:
        result = openai_api.query_with_retries(
            query, 
            max_tokens=256)
        return {'title': query, 'description': result}
    except Exception as e:
        print('OpenAI Exception:', query, '----', e)
        return None 
    
def query_wikipedia(query):
    try:
        result = wikipedia_api.search(query)
        if 'could not find' in result.lower():
            print('No result by Wikipedia:', query, '-----', result)
            return None
        return {'title': query, 'description': result}
    except:
        print('Wikipedia Exception:', query, '-----', e)
        return None 
    
def query_all_tools(query, combined_query):
    tool_outputs = []
    sources = ['chatgpt', 'internet', 'wikipedia']
    for source in sources:
        if source == 'chatgpt':
            result = query_openai(combined_query)
        elif source == 'internet':
            result = internet_search(combined_query)
        elif source == 'wikipedia' and query is not None and len(query)>0: 
            result =  query_wikipedia(query)
        if not invalid_response(result): 
            result['source'] = source 
            tool_outputs.append(result) 
   
    if len(tool_outputs)==0:
        print("No found result in all search:", combined_query)
    
    return tool_outputs 
    
def invalid_response(response):
    if response is None:
        return True
    if 'description' in response:
        description = response['description']
    else:
        return True 
    if len(description.strip())==0:
        return True
    for seq in refusal_seqs:
        if seq in description.lower():
            return True
    return False 

def code_search(query, snippet=None):
    if snippet is not None and len(snippet.strip())>0: 
        combined_query = 'Code context:\n```{}\n```'.format(snippet) + '\nQuery: ' + query
    else:
        combined_query = 'Provide critical and useful information about the following: ' + query 
    
    outputs = query_all_tools(query, combined_query)
    
    return outputs  

def code_review(query=None, code=None):
    snippet = code 
    if query is None and snippet is None:
        return None
    
    combined_query = ''
    if snippet is not None and len(snippet.strip())>0:         
        execution_result = run_code(snippet)
        if len(execution_result.strip())==0:
            execution_result = 'the code is compiled successfully without any error.'
        combined_query += 'Code context:\n```{}\n```'.format(snippet) + \
            '\nCode output: ' + execution_result 
        if query is not None:
            combined_query += '\nQuery: ' + query    
    elif query is not None: 
        combined_query += 'Provide critical and useful information about the following: ' + query 
    
    outputs = query_all_tools(query, combined_query)
    
    return outputs     
    