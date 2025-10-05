from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from agaie.agent.tools.tools import calculator, make_knowledge_base_search_tool
from agaie.observability.observability import instrument_tool

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

def create_agent(query_engine):

    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "The user will choose a certain topic or stock to query you about it. Please inspect the "
        "Knowledge Base to update and complete your knowledge on said topic. After that, the user may keep on querying you about "
        "different things related to that very thing, keep on helping them."
        "When using the knowledge_base_search tool to inspect context related to the user's query, please make sure that you"
        "Cite text and URLs (if present) at the end.\n\n"),
        ("human", "{input}")
    ])

    kb_search_tool = make_knowledge_base_search_tool(query_engine)
    # Instrument tools so we get tool latency and error counters
    tools = [
        instrument_tool(calculator), 
        instrument_tool(kb_search_tool)
    ]

    agent =  prompt | llm.bind_tools(tools, tool_choice='auto')

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    return agent_executor, llm.model_name
    
    
