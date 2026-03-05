import json
from langchain.tools import tool
from e2b_code_interpreter import Sandbox
from config.logger import get_logger

logger = get_logger(__name__)

@tool
async def e2b_run_code(python_code: str) -> str:
    """
    Execute Python code in a secure E2B sandbox environment.

    Use this for custom Python scripts, data analysis, visualizations,
    or any computation that doesn't require Anthropic-specific features like skills.
    Supports Jupyter Notebook syntax, matplotlib, pandas, numpy, etc.

    Args:
        code: Python code to execute

    Returns:
        JSON string with execution results including stdout, stderr, and outputs
    """
    logger.info(f"Executing code in E2B sandbox: {python_code[:100]}...")

    try:
        sbx = Sandbox.create()
        execution = sbx.run_code(python_code)

        logger.info(f"Execution completed. Stdout: {len(execution.logs.stdout)} chars, "
                   f"Stderr: {len(execution.logs.stderr)} chars")

        result = {
            "success": True,
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "results": [str(r) for r in execution.results] if execution.results else []
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"E2B execution error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e)
        }, indent=2)

    finally:
        sbx.kill()
