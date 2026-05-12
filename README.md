# 简介

微信自动聊天助手，借助AI帮助回复微信消息

# 🚀 快速开始

### 1. 环境要求

- Python 3.12
- Windows 10/11
- 微信桌面版3.9

### 2. 安装依赖

```bash
# 使用 uv 安装
uv sync

# 或手动安装
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

MODEL_NAME必须要多模态的大模型，支持图片识别

```env
# LM Studio API 配置
API_BASE=http://localhost:1234/v1
API_KEY=lm-studio
MODEL_NAME=qwen3-vl-8b-instruct
```

### 4. 导入聊天记录

聊天记录的功能是用于搜索历史聊天是否有相同问答，用于辅助ai生成答复。若不想导出聊天记录，使用历史回溯，可以跳过这一步。

准备工作：

根据https://github.com/Sunldon/fork-WeChatMsg.git项目导出qq聊天记录，会生成wechat_cleaned.md，放置于当前项目根目录下

首次运行时，需要下载向量数据库bge-m3，下载到当前目录下models/bge-m3，可以从ModelScope或者hf-mirror.com下，下面的命令也可以下载

```
uv run python download_models.py
```

执行命令，将当前目录下的wechat_cleaned.md 导入向量数据库

```
uv run python parse_wechat.py --parse
```

### 5. 生成person人物性格

使用项目https://github.com/agenmod/immortal-skill.git 生成人物性格，放置在person目录下，当前目录结构

```

    目录: autoWechat\person
Mode                 LastWriteTime         Length Name                                                 
----                 -------------         ------ ----                                                 
-a----         2026/4/23     15:47           2407 interaction.md                                       
-a----         2026/4/23     15:48           1916 memory.md                                            
-a----         2026/4/23     15:47           2329 personality.md                                       
-a----         2026/4/23     15:48           3497 SKILL.md  
```

如果第4步未执行，而immortal-skill需要依赖聊天记录蒸馏自己，为了使用特定人格进行回复，可以使用现成的[永生.skill — 把任何人从聊天记录里蒸馏出来](https://agenworld.com/market)，使用各种名人的性格来进行回复消息。

或者使用nuwa-skill生成SKILL.md，放再person目录下

### 6. 运行程序

**启动说明：**

需要提前打开微信，最小化在任务栏，而不是隐藏在右下角

![1778597283467](image/README/1778597283467.png)

配置联系人

在 `config.yaml` 中配置要监控的联系人：

```yaml
wechat:
  contacts:
    - "张三"
    - "李四"
```

修改姓名，让AI认识到用户姓名：

```
  user:
    name: "张三"       # 用户姓名
```

若没有导入聊天记录，不依赖聊天记录启动：

```bash
# 运行主程序
uv run python auto_wechat.py --no-history

# 或直接运行
python auto_wechat.py --no-history
```

有导入聊天记录，启动：

```
# 运行主程序
uv run python auto_wechat.py --no-history

# 或直接运行
python auto_wechat.py --no-history
```

若运行出现问题，可以按照下面的模块说明，逐步排查

若觉得回复风格不行，可以修改ai_analyse.py的create_system_prompt提示词。

## 模块说明

### 1. operate_wechat.py - 微信窗口操作模块

提供微信窗口控制、截图、消息发送等功能，指定联系人为张三

```
uv run python operate_Wechat.py --target 张三
```

查看当前目录下是否有debug_last_msg.png生成，该图片为与指定人员的微信聊天对话截图，不包含侧边栏的联系人.
如果聊天框截图不完整，或者是无法命中指定联系人，可以调整config.yaml的配置，如下所示

```
# 聊天区域截取配置
capture:
  left_offset: 300      # 左侧偏移量（跳过左侧联系人栏）
  top_offset: 90        # 顶部偏移量（跳过标题栏）
  bottom_offset: 205   # 底部偏移量（跳过输入栏）
search_position:
  x_offset: 150       # 搜索框相对于窗口左边的水平偏移
  y_offset: 40        # 搜索框相对于窗口顶部的垂直偏移
```

### 2. ai_analyse.py - AI 分析与对话模块

```
uv run python ai_analyse.py
```

根据debug_last_msg.png，让ai进行分析图片中的聊天内容，会生成以下内容：

```
当前消息结构:
{'text': 'xx', 'sender': 'self'}
{'text': 'yyy', 'sender': 'self'}
{'text': 'aaa', 'sender': 'other'}
{'text': 'bbbb', 'sender': 'self'}
```

若最后一条消息sender是other，则会将消息发送给ai，让ai进行回复

## 数据流

```
屏幕截图 → operate_wechat.py (capture_chat_area)
         ↓
      ai_analyse.py (ai_get_messages)
      调用 Qwen3-VL (LM Studio) 解析聊天界面 → JSON 消息列表
         ↓
      parse_wechat.py (query_context)
      ChromaDB 检索相关历史回复作为上下文
         ↓
      ai_analyse.py (chat_with_digital_twin)
      调用 Qwen3-VL 生成符合人格的回复
         ↓
      operate_wechat.py (send_message_chinese)
      剪贴板 + Ctrl+V 发送文字
```

## ⚙️ 配置文件说明

### config.yaml

主配置文件，采用 YAML 格式。

```yaml
# 微信自动化配置
wechat:
  window:
    width: 1000
    height: 800

  # 用户配置（本人信息）
  user:
    name: "张三"       # 用户姓名

  # 联系人列表
  contacts:
    - "李四"
    - "王五"

  # 消息监听配置
  monitor:
    base_sleep_time: 20  # 基础休眠时间（秒）
    max_sleep_time: 600  # 最大休眠时间（秒）
    retry_attempts: 3  # 消息发送重试次数

  # 聊天区域截取配置
  capture:
    left_offset: 300      # 左侧偏移量（跳过左侧联系人栏）
    top_offset: 90        # 顶部偏移量（跳过标题栏）
    bottom_offset: 205   # 底部偏移量（跳过输入栏）
# LM Studio / Qwen3-VL 模型配置
model:
  api_base: "http://localhost:1234/v1"
  api_key: "lm-studio"
  model_name: "qwen3-vl-8b-instruct"  # 注意：LangChain ChatOpenAI 使用 "gpt-4" 类 model name 会被忽略，实际以 api_base 为准
  max_tokens: 51200
  temperature: 0.85

# 调试配置
debug:
  screenshot_path: "debug_last_msg.png"
```

### .env 环境变量

优先级高于 config.yaml 的环境变量配置。

```env
# LM Studio API 配置
API_BASE=http://localhost:1234/v1
API_KEY=lm-studio
MODEL_NAME=qwen3-vl-8b-instruct
```

**说明：** `.env` 中的配置会覆盖 `config.yaml` 中的对应配置。

### 配置优先级

1. `.env` 环境变量（最高）
2. `config.yaml` 文件配置
3. 代码默认值（最低）

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
