from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agaie.agent.tools.tools import calculator, make_knowledge_base_search_tool, final_answer

from dotenv import load_dotenv

"""
- Best practice in production: use explicit lanes (ingest + query) with automatic retrieval, 
    and optionally expose the same retriever as a tool for follow-up hops.

- Why: determinism, observability, cost/SLO control, and easier testing.

- Why LangGraph over hand-rolled FastAPI logic: you get a real state machine (typed state, 
    conditional edges, retries, timeouts, parallel steps) and built-in tracing, instead of 
    re-implementing orchestration inside HTTP handlers.
"""

# TODO: first think of the "app / entry layer", and how it would connect to the ingest / query lanes (e.g. a FastAPI layer for ingestion / query)

load_dotenv()

def create_agent(query_engine, max_steps: int = 6, model: str = "gpt-4o-mini", temperature: float = 0.0):

    llm = ChatOpenAI(temperature=temperature, model=model, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "The user will choose a certain topic or stock to get to know more about. Please inspect the "
        "Knowledge Base to update and complete your knowledge on said topic. After that, the user may keep on querying you about "
        "different things related to that very thing, keep on helping them. You will have access to the tools below:\n"
         "- `knowledge_base_search`: retrieve relevant, factual information.\n"
         "- `calculator`: compute numeric expressions when needed.\n"
         "- `final_answer`: call this when you are ready to give your final answer.\n\n"
        "When using the knowledge_base_search tool to inspect context related to the user's query, please make sure that you"
        "Cite text and URLs (if present) at the end.\n\n"
        "Finally, When you believe you have enough context, call `final_answer(content=...)`. "
        "After calling it, respond directly with that text and do not invoke more tools."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    kb_search_tool = make_knowledge_base_search_tool(query_engine)
    tools = [calculator, kb_search_tool, final_answer]

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_steps,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )

    return agent_executor, llm.model_name
    
    
