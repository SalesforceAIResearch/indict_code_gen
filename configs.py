from prompts import *
from agents import AgentStrategy
import tools as tool_functions

strategy_mapping = {}
for s in AgentStrategy:
    strategy_mapping[s.value] = s.name 

model_mapping = {
    'codellama-7b-instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'codellama-13b-instruct': 'codellama/CodeLlama-13b-Instruct-hf',
    'codellama-34b-instruct': 'codellama/CodeLlama-34b-Instruct-hf',
    'codellama-70b-instruct': 'codellama/CodeLlama-70b-Instruct-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama3-8b-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct', 
    'llama3-70b-instruct': 'meta-llama/Meta-Llama-3-70B-Instruct', 
    'commandr': 'CohereForAI/c4ai-command-r-v01',
    'gpt4': 'gpt-4-32k',
    'gpt3.5': 'gpt-3.5-turbo'
}

agent_configs = {    
    "indict_llama":{
        'actor_prompt': actor_prompt,
        'safety_critic_prompt': safety_critic_prompt,
        'helpful_critic_prompt': helpful_critic_prompt,
        'summary_critic_prompt': summary_critic_prompt,
        'summary_critic_prompt_posthoc': summary_critic_prompt_posthoc, 
        'tool_prompt': query_tool_prompt,
        'tool_prompt_code': query_tool_prompt_with_code, 
        'tool_prompt_posthoc': query_tool_prompt,
    },
    "indict_commandr":{
        'actor_prompt': actor_prompt,
        'safety_critic_prompt': safety_critic_prompt,
        'helpful_critic_prompt': helpful_critic_prompt,
        'summary_critic_prompt': summary_critic_prompt,
        'summary_critic_prompt_posthoc': summary_critic_prompt_posthoc,
        'tool_prompt': query_tool_use_prompt,
        'tool_prompt_posthoc': query_tool_use_prompt_posthoc,
    }
}

tool_definitions = {
  'codesearch': {
    "name": "code_search",
    "description": "Returns a list of relevant document snippets for a textual query about a relevant code snippet retrieved from the internet",
    "parameter_definitions": {
      "query": {
        "description": "Query about the code snippet to search the internet with",
        "type": 'str',
        "required": True
      },
      "snippet": {
        "description": "A short code snippet that the query is based on",
        "type": 'str',
        "required": True
      }
    }
  },
  'codereview': {
    "name": "code_review",
    "description": "Returns a list of relevant document snippets for a textual query about a relevant code (and its corresponding execution results) retrieved from the internet",
    "parameter_definitions": {
      "query": {
        "description": "Query about the code (and its corresponding execution outputs) to search the internet with",
        "type": 'str',
        "required": True
      },
      "code": {
        "description": "A short code to be executed and for the query to be based on. The code should be independent and containt print() statements in important positions to debug variable values.",
        "type": 'str',
        "required": True
      }
    }
  }
}

tool_function_mapping = {
    'code_search': tool_functions.code_search, 
    'code_review': tool_functions.code_review
}