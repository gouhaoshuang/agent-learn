"""列目录工具。只允许访问 data/ 子树 —— codelens 自己的实现代码不是用户想问的内容。"""

from pathlib import Path

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
MAX_ENTRIES = 200


def _guard_under_data(abs_path: Path) -> str | None:
    """拦截 data/ 外的路径。返回错误字符串表示拦下了，None 表示放行。"""
    try:
        abs_path.resolve().relative_to(DATA_ROOT.resolve())
        return None
    except ValueError:
        return (f"(error) path must be under data/ (目标代码与文档的根目录)；"
                f"got: {abs_path}。"
                f"要看整体结构请用 list_files('data')。")


@tool
def list_files(path: str = "data", max_depth: int = 2) -> str:
    """列出目录下的子目录/文件（类似 ls -R），回答"有哪些代码 / 哪些文档可查"类问题。

    path 相对 codelens/ 根目录，**必须在 data/ 下**（目标代码 data/code/、目标文档 data/docs/）。
    若想看整体目录概况，请用 list_files("data")；想深入某个子目录，再传子路径。
    max_depth 限制递归深度（默认 2）。
    """
    p = Path(path)
    abs_path = p if p.is_absolute() else (PROJECT_ROOT / p)

    err = _guard_under_data(abs_path)
    if err is not None:
        return err
    if not abs_path.exists():
        return f"(error) path not found: {abs_path}"
    if not abs_path.is_dir():
        return f"(error) not a directory: {abs_path}"

    base_depth = len(abs_path.parts)
    lines = [f"{abs_path}/"]
    count = 0
    for sub in sorted(abs_path.rglob("*")):
        rel_parts = sub.relative_to(abs_path).parts
        if any(p.startswith(".") for p in rel_parts):  # 跳隐藏
            continue
        depth = len(sub.parts) - base_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        marker = "/" if sub.is_dir() else ""
        lines.append(f"{indent}{sub.name}{marker}")
        count += 1
        if count >= MAX_ENTRIES:
            lines.append(f"... (truncated at {MAX_ENTRIES} entries)")
            break
    return "\n".join(lines)
