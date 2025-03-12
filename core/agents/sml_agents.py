import argparse
import os

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    GradioUI,
    HfApiModel,
    LiteLLMModel,
    Tool,
    load_tool,
    tool,
)

# @tool
# def run_browser_agent(task:str)->None|str:
#     '''This tools uses browser to perform actions in the web browser, thik of this browser agent, it requires a task(string)
#     Args:
#         task: task to perform, be discriptive about the task
#     Returns:
#         result: result of the task

#     '''
#     try: 
#         from browser_use import (
#             Agent,
#             Browser,
#             BrowserConfig,
#             BrowserContextConfig,
#         )
#         from browser_use.browser.context import BrowserContext
#         from dotenv import load_dotenv

#         # from langchain_openai import ChatOpenAI
#         from langchain_google_genai import ChatGoogleGenerativeAI
#     except ImportError:
#         raise ImportError("Please install browser_use and langchain_openai or langchain google genai first. use `pip install browser_use langchain_openai`")

#     import asyncio
#     load_dotenv()

#     # Basic configuration
#     config = BrowserConfig(
#         headless=False,
#         disable_security=True
#     )
    

#     context_config = BrowserContextConfig(
#         wait_for_network_idle_page_load_time=0.1,
#         browser_window_size={'width': 1280, 'height': 1100},
#         locale='en-US',
#         user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
#         highlight_elements=True,
#         viewport_expansion=500,
#         # allowed_domains=['google.com', 'wikipedia.org'],
#     )

#     browser = Browser(config=config)
#     context = BrowserContext(browser=browser, config=context_config)

#     # llm = ChatOpenAI(model="gpt-4o")
#     llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
#     async def main():
#         agent = Agent(
#             task=task,
#             llm=llm,
#     		browser_context=context,

#         )
#         result = await agent.run()
#         return result.final_result()

#     out =asyncio.run(main())
#     return out

# tools
# image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)




# web_action_agent = CodeAgent(
#     tools=[run_browser_agent],
#     model=model,
#     name="web_search",
#     description="Performs action on the web browser, can control browser. Give it your query as a task."
# )




def get_agent(model_id:str):
    """Get an agent with the specified model ID."""
    try:
        from smolagents import (
            CodeAgent,
            DuckDuckGoSearchTool,
            LiteLLMModel,
        )
    except ImportError:
        raise ImportError(
            "smolagents is not installed. Install it with: pip install 'llm-cli[agents]'"
        )

    model = LiteLLMModel(model_id=model_id)
    web_search_agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        name="web_search_agent",
        description="Runs simple web searches for you. Give it your query as an argument."
    )
    
    python_code_agent = CodeAgent(
    tools=[],
    name="python_coder",
    description="Does coding and runs python code",
    model=model,
    add_base_tools=True,
    )
    
    bash_code_agent = CodeAgent(
        tools=[],
        name="bash_coder",
        description="Solves bash problems, writes/runs bash code",
        model=model,
        add_base_tools=True,
    )

    
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[
            web_search_agent, 
            # web_action_agent, 
            python_code_agent,
            bash_code_agent,
            ]  # image agent now optional
    )
    return manager_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent CLI")
    parser.add_argument("task", type=str, nargs="?", help="Task for the agent to process")
    parser.add_argument("--gui", action="store_true", help="Launch Gradio UI in browser")
    parser.add_argument("--include-image", action="store_true", help="Include image generation agent")
    args = parser.parse_args()
    
    model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    hf_token = os.environ["HF_TOKEN"] 
    model = HfApiModel(model_id=model_id, token=hf_token)
    model = HfApiModel()

    manager_agent = get_agent(model_id)


    if args.include_image:
        image_generation_tool = Tool.from_space(
            "black-forest-labs/FLUX.1-schnell",
            name="image_generator",
            description="Generate an image from a prompt"
        )

        # Define image_generation_agent (optional)
        image_generation_agent = CodeAgent(
            tools=[image_generation_tool],
            model=model,
            name="image-generation-agent-1",
            description="Generates image from given prompt. Give it your query as an argument."
        )
        manager_agent.managed_agents.append(image_generation_agent)

    if args.gui:
        GradioUI(manager_agent).launch(inbrowser=True)
    elif args.task:
        result = manager_agent.run(args.task)
        print(result)
    else:
        print("Please provide a task or use --gui to launch the GUI.")