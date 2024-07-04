import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
import numpy as np 

import pdb
import copy 
import json

from llm import OPENAI 
from prompts import * 
from util import extract_tools, extract_code, format_step, parse_action 
import tools as tool_functions 

from langchain.prompts import PromptTemplate

MAX_TOKENS=1024

class AgentStrategy(Enum):
    INDICT_LLAMA = 'indict_llama'
    INDICT_COMMANDR = 'indict_commandr'

class Agents:
    def __init__(self, 
        sample_idx: int,   
        question: str,
        system_prompt = None, 
        actor_prompt: PromptTemplate = None, 
        safety_critic_prompt: PromptTemplate = None, 
        helpful_critic_prompt: PromptTemplate = None, 
        summary_critic_prompt: PromptTemplate = None, 
        summary_critic_prompt_posthoc: PromptTemplate = None, 
        tool_prompt: PromptTemplate = None, 
        tool_prompt_posthoc: PromptTemplate = None, 
        tool_prompt_code: PromptTemplate=None, 
        action_llm: OPENAI = OPENAI(
                                model_name="gpt-3.5-turbo",
                                api_key=os.environ['OPENAI_API_KEY']),
        critic_llm: OPENAI = OPENAI(
                                model_name="gpt-3.5-turbo", 
                                api_key=os.environ['OPENAI_API_KEY']),
        action_prompt_header: str = '',
        critic_tool = None, 
        critic_tool_posthoc = None,
        prev_trial=None,
        task=None, 
        tokenizer=None, 
        programming_language=None,
        code_before='',
        ) -> None:
        
        self.sample_idx = sample_idx
        self.question = question
        
        self.system_prompt = system_prompt
        if task == 'promptinject': 
            self.question_only = self.question
            self.question = 'System rule: ' + self.system_prompt + '\n' + self.question 
        
        self.actor_prompt = actor_prompt
        
        self.safety_critic_prompt = safety_critic_prompt
        self.helpful_critic_prompt = helpful_critic_prompt
        self.summary_critic_prompt = summary_critic_prompt
        self.summary_critic_prompt_posthoc = summary_critic_prompt_posthoc
        
        self.tool_prompt = tool_prompt
        self.tool_prompt_posthoc = tool_prompt_posthoc
        self.tool_prompt_code = tool_prompt_code
                
        self.action_prompt_header = action_prompt_header        
        
        self.action_llm = action_llm
        self.critic_llm = critic_llm
        
        self.critic_tool = critic_tool 
        self.critic_tool_posthoc = critic_tool_posthoc

        self.prev_trial = json.load(open(prev_trial, 'r')) if (prev_trial is not None and os.path.exists(prev_trial)) else None 
        if self.prev_trial is not None: assert self.prev_trial['sample_idx'] == self.sample_idx 
       
        self.num_actions = 1 
        self.critic_rounds = 1
        self.num_tool_queries = 1 
        
        self.task = task 
        self.tokenizer = tokenizer 
        self.programming_language = programming_language
        self.code_before = code_before        
        
        self.reset()
        
    def run(self, strategy: AgentStrategy):
        self.strategy = strategy 
        self.reset()
        self.step()
        
        output = {'sample_idx': self.sample_idx, 
                  'action': self.action, 
                  'scratchpad': self.scratchpad}
        
        if self.prev_trial is not None:
            output['critic'] = self.prev_trial['critic']
            if type(output['critic'])==str:
                output['critic'] = [self.prev_trial['critic']]
            output['critic'].append(self.critic)
        else:
            output['critic'] = self.critic 

        if self.programming_language=='python':
            if self.prev_trial is not None:
                output['critic_posthoc'] = self.prev_trial['critic_posthoc']
                if type(output['critic_posthoc'])==str:
                    output['critic_posthoc'] = [self.prev_trial['critic_posthoc']]
                output['critic_posthoc'].append(self.critic_posthoc)
            else:
                output['critic_posthoc'] = self.critic_posthoc
            output['mid_action'] = self.mid_action

        output['initial_action'] = self.initial_action
        output['safety_critics'] = self.safety_critics
        output['helpful_critics'] = self.helpful_critics

        if self.programming_language=='python':
            output['safety_critics_posthoc'] = self.safety_critics_posthoc
            output['helpful_critics_posthoc'] = self.helpful_critics_posthoc

        output['safety_tool_output'] = self.safety_tool_output
        output['helpful_tool_output'] = self.helpful_tool_output 
        output['critic_scratchpad'] = self.critic_scratchpad

        if self.programming_language=='python':
            output['safety_tool_output_posthoc'] = self.safety_tool_output_posthoc
            output['helpful_tool_output_posthoc'] = self.helpful_tool_output_posthoc 
            output['critic_scratchpad_posthoc'] = self.critic_scratchpad_posthoc

        return output
        
    def step(self) -> None:
        self.scratchpad += self.action_prompt_header 
        
        # Initial Act 
        if self.prev_trial is None: 
            self.action = self.prompt_agent(self.action_llm, self.actor_prompt)   
        else:
            self.action = self.prev_trial['action']
                            
        # Preemptive Critic 
        self.scratchpad = '\nSolution: ' + self.action_prompt_header                 
        self.scratchpad += self.action 
        self.critic = self.perform_critic_debate(answer=self.action, max_steps=self.critic_rounds)

        # Act again 
        self.scratchpad = ''
        self.scratchpad += '\nInitial Solution: ' + self.action
        self.scratchpad += '\nCritic: ' + self.critic + \
            '\nImproved Solution: ' + self.action_prompt_header
        self.initial_action = self.action 
        self.action = self.prompt_agent(self.action_llm, self.actor_prompt, stop_seqs=['\nCritic:'])
        self.scratchpad += self.action

        # Posthoc Critic 
        if self.programming_language=='python':
            action_execution = tool_functions.run_code(self.code_before + extract_code(self.action))
            if len(action_execution)>0:
                self.action_execution = action_execution
            else:
                self.action_execution = 'Solution is compiled successfully without any error.'
            self.scratchpad += '\nObservation: ' + self.action_execution
            critic = self.perform_critic_debate(answer=self.action, 
                                                          max_steps=self.critic_rounds, 
                                                          posthoc=True)
            self.critic_posthoc = critic
            
            # Act again after Posthoc 
            self.scratchpad = ''
            self.scratchpad += '\nInitial Solution: ' + self.initial_action
            self.scratchpad += '\nCritic: ' + self.critic + \
                '\nFirst Improved Solution: ' + self.action + \
                '\nCritic: ' + self.critic_posthoc + \
                '\nSecond Improved Solution: ' + self.action_prompt_header 
            self.mid_action = self.action 
            self.action = self.prompt_agent(self.action_llm, self.actor_prompt, stop_seqs=['\nCritic:'])
            self.scratchpad += self.action
            
    def perform_critic_debate(self, max_steps=1, prefix='', answer=None, posthoc=False):
        if posthoc:
            posthoc_suffix = '_posthoc'
        else:
            posthoc_suffix = ''
        setattr(self, prefix + "safety_critics" + posthoc_suffix, [])
        setattr(self, prefix + "helpful_critics" + posthoc_suffix, []) 
        safety_critic_prompt = getattr(self, prefix + 'safety_critic_prompt')
        helpful_critic_prompt = getattr(self, prefix + 'helpful_critic_prompt')
        summary_critic_prompt = getattr(self, prefix + 'summary_critic_prompt' + posthoc_suffix)
        tool_prompt = getattr(self, prefix + 'tool_prompt' + posthoc_suffix)
        critic_tool = getattr(self, prefix + 'critic_tool' + posthoc_suffix)
        
        self.scratchpad = ''
        if self.prev_trial is not None:
            self.scratchpad += '\nThe following critic(s) provide some analysis of system safety and correctness of previous solution(s). Use them as reference only to improve and update your critic based on the above latest solution.'
            
            prev_critics = self.prev_trial[prefix + 'critic']
            if type(prev_critics)==str: 
                prev_critics = [prev_critics]
                
            if prefix + 'critic_posthoc' in self.prev_trial:
                prev_posthoc_critics = self.prev_trial[prefix + 'critic_posthoc']
                if type(prev_posthoc_critics)==str:
                    prev_posthoc_critics = [prev_posthoc_critics]                    
                    
                for pc, ppc in zip(prev_critics, prev_posthoc_critics):
                    self.scratchpad += '\nPast Critic: ' + pc
                    self.scratchpad += '\nPast Critic: ' + ppc
            else:
                for pc in prev_critics:
                    self.scratchpad += '\nPast Critic: ' + pc
                
        if posthoc: 
            self.scratchpad += '\nThe following critic(s) provide some analysis of system safety and correctness of previous solution(s). Use them as reference only to improve and update your critic based on the above latest solution.'
            self.scratchpad += '\nPast Critic: ' + self.critic
            
        setattr(self, prefix + "safety_tool_output" + posthoc_suffix, [])
        setattr(self, prefix + "helpful_tool_output" + posthoc_suffix, []) 
        
        if posthoc: 
            self.scratchpad += '\nThe following provides some observation(s) when executing the current solution. If these observations are relevant, you might use them as reference (if needed) to improve and update your critic:'
            if self.task == 'humaneval' and len(self.example_test_cases)>0:
                self.scratchpad += '\nTest Cases:\n' + '\n'.join(self.example_test_cases)
            self.scratchpad += '\nCurrent Solution Observation: ' + self.action_execution
            
        for step in range(max_steps):
            ## STEP 1A: SAFETY CRITIC
            self.scratchpad += '\nSafety Critic: '
            if step > 0 or self.prev_trial is not None: 
                self.scratchpad += 'based on the above discussion, here is my updated analysis of solution safety: '
            else:
                self.scratchpad += 'here is my analysis of solution safety: '
            safety_critic = self.prompt_critic_agent(self.critic_llm, safety_critic_prompt, 
                                                max_tokens=128, answer=answer)
            
            ## STEP 1B: SAFETY CRITIC TOOL 
            curr_scratchpad = self.scratchpad 
            query_tool_output_str, tool, tool_output = self.query_tools(safety_critic, 
                                                                            answer=answer, 
                                                                            tool_prompt=tool_prompt,
                                                                            critic_tool=critic_tool)
            if len(query_tool_output_str.strip())>0:
                safety_critic += '\nSupporting Fact(s) for Safety: ' + query_tool_output_str
            self.scratchpad = curr_scratchpad 
            getattr(self, prefix + "safety_tool_output" + posthoc_suffix).append({'tool': tool, 'output': tool_output})
            
            getattr(self, prefix + "safety_critics" + posthoc_suffix).append(safety_critic)
            self.scratchpad += safety_critic + '\nCorrectness Critic: '
            
            # STEP 2A: HELPFULNESS CRITIC
            if step > 0 or self.prev_trial is not None: 
                self.scratchpad += 'based on the above discussion, here is my updated analysis of solution correctness: '
            else:
                self.scratchpad +=  'here is my analysis of solution correctness: '
            helpful_critic = self.prompt_critic_agent(self.critic_llm, helpful_critic_prompt, 
                                                max_tokens=128, answer=answer)
            
            # STEP 2B: HELPFULESS CRITIC TOOL 
            curr_scratchpad = self.scratchpad 
            query_tool_output_str, helpful_tool, helpful_tool_output = self.query_tools(safety_critic, 
                                                                                            answer=answer,
                                                                                            tool_prompt=tool_prompt,
                                                                                            critic_tool=critic_tool)

            if len(query_tool_output_str.strip())>0:
                helpful_critic += '\nSupporting Fact(s) for Correctness: ' + query_tool_output_str 
            self.scratchpad = curr_scratchpad 
            getattr(self, prefix + "helpful_tool_output" + posthoc_suffix).append({'tool': helpful_tool, 'output': helpful_tool_output})
            
            getattr(self, prefix + "helpful_critics" + posthoc_suffix).append(helpful_critic)
            self.scratchpad += helpful_critic
            
        critic_summary = self.prompt_agent(self.critic_llm, summary_critic_prompt, max_tokens=1024)
        setattr(self, prefix + "critic_scratchpad" + posthoc_suffix, self.scratchpad)
        
        return critic_summary 
    
    def query_tools(self, critic, answer, tool_prompt, critic_tool):            
        self.scratchpad = 'Analysis:' + critic
        if self.strategy == AgentStrategy.INDICT_LLAMA:
            tool = self.prompt_critic_agent(self.critic_llm, tool_prompt, max_tokens=64, answer=answer)
            parsed_tool = None
            for line in tool.split('\n'):
                parsed_tool = parse_action(line)
                if parsed_tool is not None: 
                    break
            query = "" 
            if parsed_tool is not None: 
                _, query = parsed_tool
                
            generated_query_code = self.prompt_critic_agent(self.critic_llm, self.tool_prompt_code, 
                                                  max_tokens=128, answer=answer, query = query)
            query_code = extract_code(generated_query_code)
            
            if critic_tool[0]['name'] == 'code_search':
                tool_selection = {
                    'tool_name': critic_tool[0]['name'],
                    'parameters': {
                        'query': query,
                        'snippet': query_code
                    }
                }
            elif critic_tool[0]['name'] == 'code_review': 
                tool_selection = {
                    'tool_name': critic_tool[0]['name'],
                    'parameters': {
                        'query': query,
                        'code': query_code
                    }
                }
            tool_selections = [tool_selection]
            
        elif self.strategy == AgentStrategy.INDICT_COMMANDR:
            context = self._build_critic_agent_prompt(tool_prompt, '', answer)
            conversation = [{"role": "user", "content": context}]
            tool_use_prompt = self.tokenizer.apply_tool_use_template(
                conversation,
                tools=critic_tool,
                tokenize=False,
                add_generation_prompt=True,
            )
            tool_selections = self.critic_llm.query_with_retries(tool_use_prompt, max_tokens=512)
            tool_selections = extract_tools(tool_selections)

        tool_outputs = [] 
        tool_output_str = ''
        num_queries = 0 
        for tool in tool_selections:
            try:
                outputs = getattr(tool_functions, tool['tool_name'])(**tool['parameters'])
                num_queries += 1 
                for output in outputs:
                    if output is not None and len(output)>0:
                        current_output_str = ''
                        if 'title' in output and output['title']:
                            current_output_str += output['title']
                        if 'description' in output and output['description']:
                            current_output_str += ' - ' + output['description']
                        if len(current_output_str)>0: 
                            tool_output_str += '\nSupporting Fact: ' + current_output_str
                            tool_outputs.append(output)
                if num_queries == self.num_tool_queries:
                    break 
            except Exception:
                continue 
        return tool_output_str, tool_selections, tool_outputs
        
    def reset(self) -> None:
        self.scratchpad: str = ''

    def prompt_agent(self, llm_module, prompt_template, max_tokens=1024, stop_seqs=[], 
                     num_outputs=1, main_action=True) -> str:
        prompt = self._build_agent_prompt(prompt_template, main_action)
        if main_action and self.task == 'promptinject':
            output = llm_module.query_with_system_prompt_with_retries(
                self.system_prompt, prompt, max_tokens=max_tokens, stop_seqs=stop_seqs, num_outputs=num_outputs)
        else:
            output = llm_module.query_with_retries(
                prompt, max_tokens=max_tokens, stop_seqs=stop_seqs, num_outputs=num_outputs)
        return format_step(output)
    
    def _build_agent_prompt(self, prompt_template, main_action) -> str:
        if main_action and self.task == 'promptinject':
            question = self.question_only 
        else:
            question = self.question
        if 'question' in prompt_template.input_variables:
            return prompt_template.format(
                                question = question,
                                scratchpad = self.scratchpad)
        return prompt_template.format(scratchpad = self.scratchpad) 
    
    def prompt_critic_agent(self, llm_module, prompt_template, fewshots='', max_tokens=1024, stop_seqs=[], answer=None, query=None) -> str:
        prompt = self._build_critic_agent_prompt(prompt_template, fewshots, answer, query=None)
        return format_step(llm_module.query_with_retries(prompt, max_tokens=max_tokens, stop_seqs=stop_seqs))
    
    def _build_critic_agent_prompt(self, prompt_template, fewshots='', answer=None, query=None) -> str:
        if 'query' in prompt_template.input_variables:
            return prompt_template.format(
                            query = query, 
                            question = self.question,
                            answer = answer,
                            scratchpad = self.scratchpad)
        return prompt_template.format(
                question = self.question,
                answer = answer,
                scratchpad = self.scratchpad) 
    
    
    
