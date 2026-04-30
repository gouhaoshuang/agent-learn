"""rg / grep 封装。只允许访问 data/ 子树 —— 和 list_files / read_file 统一。"""

import subprocess
from pathlib import Path

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def _guard_under_data(abs_path: Path) -> str | None:
    try:
        abs_path.resolve().relative_to(DATA_ROOT.resolve())
        return None
    except ValueError:
        return (f"(error) path must be under data/ (目标代码与文档的根目录)；"
                f"got: {abs_path}。")


@tool
def grep_code(pattern: str, path: str = "data/code") -> str:
    """在代码目录下按正则做字面匹配，适合查找类名、函数名、宏。

    path 相对 codelens/ 根目录（默认 data/code），**必须在 data/ 下**；也可传绝对路径。
    返回最多 50 行结果；路径不存在 / 命令报错时会明确告知（不会伪装成 no match）。
    """
    # 路径解析：相对 → 相对 PROJECT_ROOT 的绝对路径
    p = Path(path)
    abs_path = p if p.is_absolute() else (PROJECT_ROOT / p)

    err = _guard_under_data(abs_path)
    if err is not None:
        return err
    if not abs_path.exists():
        return f"(error) path not found: {abs_path}"

    # 跑 rg；rg 不存在时回退 grep
    cmd_rg = ["rg", "-n", "-S", "--max-count", "3", pattern, str(abs_path)]
    cmd_grep = ["grep", "-rn", pattern, str(abs_path)]
    try:
        res = subprocess.run(cmd_rg, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        res = subprocess.run(cmd_grep, capture_output=True, text=True, timeout=10)

    # 合并 stderr：rg/grep 无命中时 returncode=1 且 stderr 为空，这是正常；
    # 但 regex 非法、权限问题等 stderr 会有内容，必须回传，否则模型看不到真实原因。
    if res.returncode not in (0, 1) or res.stderr.strip():
        err = res.stderr.strip() or f"(exit={res.returncode})"
        if not res.stdout.strip():
            return f"(error) {err}"
        out = "\n".join(res.stdout.splitlines()[:50])
        return f"{out}\n--- stderr ---\n{err}"

    out = "\n".join(res.stdout.splitlines()[:50])
    return out or "(no match)"
