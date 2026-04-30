  ① 工具调用怎么显示？
    │ A. 可展开          │ ▶ 🔧 search_docs(query="...")  │ 干净，像 ChatGPT /             │
  │ expander（推荐）   │ 点开看完整 args + 返回         │ Cursor；最接近 demo 品质       │
    ② 流式粒度？
    延续 run_cli_memory 的做法
③ thread 怎么管？   
 │ 中庸（推荐） │ 侧边栏列出 SqliteSaver 里所有        │ 实现 50                        │   
  │              │ thread_id + "New conversation" 按钮  │ 行；展示"持久化记忆"这个卖点   │   

④ 刷新浏览器后的行为？    
进入页面后从 SqliteSaver 里读当前 thread 的历史消息，回放到聊天区。   

⑤ 侧边栏放什么？    
- Sessions 列表 + New / Clear（③）
- 静态元信息：Model = deepseek-chat, Vectorstore = Chroma, Index size = 4154 chunks
- 是否展示 tool 调用统计（本轮用了几次工具、耗时）？ 不用

⑥ 文件放哪？  
  按文档建议放 codelens/app/ui.py，跑：          
 在  /data/ghs/agent-learn/codelens/ui 下面专门实现。
 可以写多个文件，分文件编写相应功能

我打算跳过的（如果你同意）   （全部跳过）
- 文件上传 → 在线重建索引（很花功夫）                                                      
- 认证 / 多用户       
- 向量库切换（Chroma vs Milvus）下拉                                                       
- 侧边栏"导出对话为 md"之类的锦上添花  