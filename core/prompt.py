system_prompt_simple = 'You are a Helpful assistant'
system_prompt_cot = "You are Intelligent, logical, anlytical, rational person. if Questions is factual answer in a single step, if questions are mathematical, reasoning, or requires stepwise thought, Think in steps. Develop a rationale"
system_prompt_rag = system_prompt_simple+" use context to answer the question if there is relevence between context and qestion"


def get_prompt(x):
    if x == 'cot':
        return system_prompt_cot
    elif x == 'rag':
        return system_prompt_rag
    else:
        return 
    
    
rag_text = """
<question>
{question}
</question>

<context>
{context}
</context>
"""


def get_formatter(x:str) -> str:
    if x == 'rag':
        return rag_text