from langchain.prompts import PromptTemplate

ACTOR_INSTRUCTION = """
{question}
{scratchpad}"""

SAFETY_CRITIC_INSTRUCTION = """You are an advanced intelligent agent with a strong focus for system security and risk management. You will be given a previous potential solution in which you were given a task to solve. In a few sentences, diagnose whether the solution is completely safe in every execution step and will not pose any potential harmful effect to the world, including any damage to cybersecurity or physical world. 

You are also required to debate and discuss with another critic who focuses on system performance and accuracy. Always advocate and focus on system security and risk management over system performance or correctness. 

Limit your critic to a few sentences. 

Task: {question}
Solution: {answer}

{scratchpad}"""

HELPFUL_CRITIC_INSTRUCTION = """You are an advanced intelligent agent with a strong focus for system performance and accuracy. You will be given a previous potential solution in which you were given a task to solve. In a few sentences, diagnose whether the solution is completely correct in every execution step and will satisfy all the requirements in the given task and pass any corner test cases. 

You are also required to debate and discuss with another critic who focuses on system security and risk management. Always advocate and focus on system performance and accuracy over system security or risk management.

Limit your critic to a few sentences. 

Task: {question}
Solution: {answer}

{scratchpad}"""

SUMMARY_CRITIC_INSTRUCTION = """You are an advanced intelligent agent with equal focuses to both system performance and security. You will be given a discussion about a system solution between two critics against system safety and correctness. In a few sentences, summarize the discussion and determine whether the solution is completely correct and secure in every execution step. The perfect solution should satisfy all the requirements in the given task while not violating any security standards or causing any harmful effect to the society or cybersecurity.  
There might be some supporting facts in the discussion between the critics. Incorporate as much as possible those details into your response to make your analysis informative and convincing.

{scratchpad}"""

SUMMARY_CRITIC_INSTRUCTION_POSTHOC = """You are an advanced intelligent agent with equal focuses to both system performance and security. You will be given a discussion about a system solution between two critics against system safety and correctness. In a few sentences, summarize the discussion and determine whether the solution is completely correct and secure in every execution step. The perfect solution should satisfy all the requirements in the given task while not violating any security standards or causing any harmful effect to the society or cybersecurity.  
There might be some supporting facts in the discussion between the critics, including relevant document snippets, code snippets and their execution results and test results. Incorporate as much as possible those details into your response to make your analysis informative and convincing to be used to improve the current initial solution.

{scratchpad}"""

QUERY_TOOL_INSTRUCTION = """You are an advanced intelligent agent with direct access to Internet. You are given a task and an example solution and relevant analysis against the solution's security or functional correctness. To improve the analysis with relevant evidence and fact, generate a relevant keyword or query to search for related information on Internet. You may also search for information that is relevant to the task or solution but is missing in the analysis. Use the following format: Search[<query or keyword>]. 

Task: {question}
Solution: {answer}

{scratchpad}

Query (in the form of Search[<query or keyword>]):"""

QUERY_TOOL_INSTRUCTION_CODE = """You are an advanced intelligent agent with direct access to Internet. You are given a task and an example solution and relevant analysis against the solution's security or functional correctness. To improve the analysis with relevant evidence and fact, a query might be provided to extract more information. To make the query more informative, extract or create a relevant short code snippet to be used together the query. If the query is empty, provide a representative code snippet that could be used to search for more information to support the analysis. 

The code snippet should be indepedent (does not refer to external operating systems, databases, repositories, or custom libraries) and limited to few lines of codes only. Use `print` or `assert` statements in the code snippets if needed (to execute and perform debugging on a code interpreter). 

Wrap the code snippet in ```. 

Task: {question}
Solution: {answer}

{scratchpad}

Query: {query}

Short code snippet in a single code block (wrap in ```):"""


QUERY_TOOL_USE_INSTRUCTION = """You are given a task and an example solution and relevant analysis against the solution's security or functional correctness. 

Read the task, solution, and analysis and find ways to improve the analysis with relevant evidence and supporting fact. You may also improve the analysis with missing information relevant to the task or solution. 

Task: {question}
Solution: {answer}

{scratchpad}

"""

QUERY_TOOL_USE_INSTRUCTION_POSTHOC = """You are given a task and an example solution and relevant analysis against the solution's security or functional correctness. Read the task, solution, and analysis and find ways to improve the analysis with relevant evidence and supporting fact. 

You also have access to a code interpreter that can execute many code snippets. Based on the solution and analysis, you can create many code snippets and unit test cases to evaluate them and support the arguments in the analysis. 

These code snippets should be indepedent (does not refer to external operating systems, databases, repositories, or custom libraries) and limited to few lines of codes only. Use `print` or `assert` statements in the code snippets if needed. 

Task: {question}
Solution: {answer}

{scratchpad}

"""


actor_prompt = PromptTemplate(
                        input_variables=["question", "scratchpad"],
                        template = ACTOR_INSTRUCTION,
                        )

safety_critic_prompt = PromptTemplate(
                        input_variables=["question", "answer", "scratchpad"],
                        template = SAFETY_CRITIC_INSTRUCTION,
                        )

helpful_critic_prompt = PromptTemplate(
                        input_variables=["question", "answer", "scratchpad"],
                        template = HELPFUL_CRITIC_INSTRUCTION,
                        )

summary_critic_prompt = PromptTemplate(
                        input_variables=["scratchpad"],
                        template = SUMMARY_CRITIC_INSTRUCTION,
                        )

summary_critic_prompt_posthoc = PromptTemplate(
                        input_variables=["scratchpad"],
                        template = SUMMARY_CRITIC_INSTRUCTION_POSTHOC,
                        )

query_tool_prompt = PromptTemplate(
                        input_variables=["question", "answer", "scratchpad"],
                        template = QUERY_TOOL_INSTRUCTION,
                        )

query_tool_prompt_with_code = PromptTemplate(
                        input_variables=["question", "answer", "scratchpad", "query"],
                        template = QUERY_TOOL_INSTRUCTION_CODE,
                        )

query_tool_use_prompt = PromptTemplate(
                        input_variables=["question", "answer", "scratchpad"],
                        template = QUERY_TOOL_USE_INSTRUCTION,
                        )


query_tool_use_prompt_posthoc = PromptTemplate(
                        input_variables=["question", "answer", "scratchpad"],
                        template = QUERY_TOOL_USE_INSTRUCTION_POSTHOC,
                        )

