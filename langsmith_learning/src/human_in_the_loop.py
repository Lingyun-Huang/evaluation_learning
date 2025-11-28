from langgraph.types import Command
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

# ---- 1. vLLM model (OpenAI-compatible API) ----
llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen3-1.7B",
)

# ---- 2. Example tools ----
@tool
def write_file_tool(content: str, filename: str):
    """Writes the given content into a file with specified filename."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Wrote file: {filename}"


search_tool = DuckDuckGoSearchRun()
# results = search_tool.invoke("Current temperature in Ottawa")
# print(results)

tools = [write_file_tool, search_tool]

# ---- 3. Create agent with Human-in-the-loop ----
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "search_tool": True,  
                "write_file_tool": {"allowed_decisions": ["approve", "reject"]}
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    # checkpointer=InMemorySaver(),
)

# ---- 4. Chat with the agent ----
# # Run the graph until the interrupt is hit.
# config = {"configurable": {"thread_id": "some_id"}} 
# result = agent.invoke(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "write 'Hello, World!' to file hello.txt.",
#             }
#         ]
#     },
#     config=config
# )
# print(result)

# ---- 5. Approve ----
# agent.invoke(
#     Command( 
#         resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
#     ), 
#     config=config # Same thread ID to resume the paused conversation
# )

# print("end")
