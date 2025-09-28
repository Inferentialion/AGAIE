from langchain_core.tools import tool
import math, numexpr


@tool
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


def make_knowledge_base_search_tool(query_engine):
    """Build the knowledge_base_search_tool, bound to a Weaviate-backed index."""
    
    @tool
    def knowledge_base_search(user_query: str, ):
        """Give the agent access to the RAG system."""
        pass


