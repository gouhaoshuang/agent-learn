# app/tools/list_files.py                                                                  
from pathlib import Path                                                                   
from langchain_core.tools import tool                                                    
                                                                                            
PROJECT_ROOT = Path(__file__).resolve().parents[2]                                       
MAX_ENTRIES = 200                  
                                                                                            
                                    
@tool                                                                                      
def list_files(path: str = "data/code", max_depth: int = 2) -> str:                      
    """列出目录下的子目录/文件，回答 '有哪些代码 / 哪些文档可查' 类问题。
    当你想要查看某个目录下的文件，文件夹时，可以使用这个命令。类似于 linux 中的 ls。
                                
    path 相对 codelens/ 根目录；max_depth 限制递归深度（默认 2）。
    典型用法：回答 "这里有哪些代码？" 时先 list_files("data/code")，                       
    看到子项后再决定是 grep / read_file 继续深入。                                       
    """                                                                                    
    p = Path(path)                    
    abs_path = p if p.is_absolute() else (PROJECT_ROOT / p)
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