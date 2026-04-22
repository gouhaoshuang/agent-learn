from pathlib import Path
import sys
import time

# 把 codelens/ 加到 sys.path，让 `from app.*` 在任意 cwd 下都能 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm import get_llm
from app.tools.search_docs import search_docs
from app.tools.grep_code import grep_code



llm = get_llm().bind_tools([search_docs, grep_code])
msg = llm.invoke("ThreadPool 类在哪个文件？")
print(msg.tool_calls)   # 应该能看到选中了 grep_code 或 search_docs