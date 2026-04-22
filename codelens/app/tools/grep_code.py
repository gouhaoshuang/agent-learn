
import subprocess
from langchain_core.tools import tool

@tool
def grep_code(pattern: str, path: str = "data/code") -> str:
    """在代码目录下按正则做字面匹配，适合查找类名、函数名、宏。返回最多 50 行结果。"""
    try:
        out = subprocess.run(
            ["rg", "-n", "-S", "--max-count", "3", pattern, path],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except FileNotFoundError:
        out = subprocess.run(
            ["grep", "-rn", pattern, path],
            capture_output=True, text=True, timeout=10,
        ).stdout
    return "\n".join(out.splitlines()[:50]) or "(no match)"