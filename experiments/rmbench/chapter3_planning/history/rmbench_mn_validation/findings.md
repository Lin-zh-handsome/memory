# Findings
- 项目 5 条 Codex 对话已同步。Put Back M0/M3 26%/100%，Swap 18%/100%，Cover 未启动。
- 服务器 8aef136，GitHub 60d0eec 为其直接后续，实验代码只有 README 差异。P1 从已发布版本隔离。
- Cover 下载重定向至官方 CDN 后读取超时，保留断点并将读取超时提高到 120 秒重试。
