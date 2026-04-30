"""按行范围读文件。只允许访问 data/ 子树 —— 和 list_files / grep_code 统一。"""

from pathlib import Path

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
MAX_LINES = 400  # 单次返回行数上限，防止把整个 httplib.h 一口气塞进 context


def _guard_under_data(abs_path: Path) -> str | None:
    try:
        abs_path.resolve().relative_to(DATA_ROOT.resolve())
        return None
    except ValueError:
        return (f"(error) path must be under data/ (目标代码与文档的根目录)；"
                f"got: {abs_path}。")


@tool
def read_file(path: str, start: int = 1, end: int | None = None) -> str:
    """读文件指定行范围，返回带行号的内容（1-based，闭区间）。

    典型用法：grep_code 找到 file:line 后，read_file(file, line-5, line+60) 读实现。
    path 相对 codelens/ 根目录，**必须在 data/ 下**。
    单次最多返回 400 行；end 省略则读 start 开始的 400 行。
    """
    p = Path(path)
    abs_path = p if p.is_absolute() else (PROJECT_ROOT / p)

    err = _guard_under_data(abs_path)
    if err is not None:
        return err
    if not abs_path.exists():
        return f"(error) file not found: {abs_path}"
    if abs_path.is_dir():
        return f"(error) path is a directory, not a file: {abs_path}"

    lines = abs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)
    s = max(1, start)
    e = min(n, (start + MAX_LINES - 1) if end is None else end, s + MAX_LINES - 1)
    if s > n:
        return f"(error) start={s} exceeds file length={n}"

    body = "\n".join(f"{i}: {lines[i-1]}" for i in range(s, e + 1))
    return f"{abs_path} [{s}-{e}]\n{body}"