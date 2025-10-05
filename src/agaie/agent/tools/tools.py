from langchain_core.tools import tool
import math, numexpr
from pydantic import BaseModel, Field


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "A mathematical expression to evaluate. "
            "Supports operators (+, -, *, /, **), parentheses, and functions "
            "like sin, cos, tan, sqrt, log, exp. Example: '2*sin(pi/3) + sqrt(5)'"
        )
    )

@tool(args_schema=CalculatorArgs)
def calculator(expression: str) -> str:
    """
    Evaluate a numeric expression safely with numexpr.

    :param expression: Algebraic expression, e.g. "2*sin(pi/3) + 7**2/5".
    :returns: Result as a string (use Decimal upstream if you need exact rounding).
    """
    # Allow only a small vocabulary; numexpr ignores Python callables by design.
    # We expose constants and map a few elementary functions by name.
    allowed = {
        "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp
    }
    # Strip whitespace; reject suspicious long inputs early if you want.
    expr = expression.strip()
    result = numexpr.evaluate(expr, global_dict={}, local_dict=allowed)
    # numexpr returns numpy scalars/arrays; coerce to plain string.
    return str(result.item() if hasattr(result, "item") else result)


class KBSearchArgs(BaseModel):
    agent_query: str = Field(
        ...,
        description=(
            "The user's natural-language question or information need. "
            "Example: 'Summarize recent news on AAPL earnings.'"
        )
    )

def make_knowledge_base_search_tool(query_engine):
    """Build the knowledge_base_search_tool, bound to a Weaviate-backed index."""
    
    @tool(args_schema=KBSearchArgs)
    def knowledge_base_search(agent_query: str):
        """Give the agent access to the RAG system."""
        
        resp = query_engine.query(agent_query)

        return resp
    
    return knowledge_base_search


class FinalAnswerArgs(BaseModel):
    content: str = Field(
        ...,
        description=(
            "The complete final answer to return to the user. "
            "Include all relevant information and citations."
        ),
    )

@tool(args_schema=FinalAnswerArgs)
def final_answer(content: str) -> str:
    """
    Explicitly signal that the agent is ready to provide the user's final answer.

    :param content: The text of the final answer, including citations if applicable.
    :return: The same content (used as a tool observation so the loop can stop).
    """
    return content