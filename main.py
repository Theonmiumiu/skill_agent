# main.py
from collections import defaultdict
from src.processor import construct_app
from src.utils.logger import logger
from langchain_core.messages import HumanMessage
from pathlib import Path


# 补丁
# ==============================================================================
# 🐒 Monkey Patch: 让 LangChain 能够识别 SiliconFlow 的 reasoning_content
# ==============================================================================
from langchain_openai.chat_models import base as langchain_openai_base

# 保存原始函数引用
_original_convert_delta = langchain_openai_base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(
        _dict, default_class
):
    # 先调用原始逻辑拿到基础 chunk
    chunk = _original_convert_delta(_dict, default_class)

    # 【核心修改】检查是否有 reasoning_content，如果有，塞进 additional_kwargs
    if "reasoning_content" in _dict:
        chunk.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]

    return chunk


# 替换掉库里的函数
langchain_openai_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk
# ==============================================================================


def main(query):
    # 构造完美符合 AgentState 结构的初始状态
    workspace = Path(__file__).parent / 'workspace'
    initial_state = {
        # 用户只需要说自然语言，不用管路径
        "messages": [HumanMessage(content=query)],

        # 物理边界：文件存在的绝对/相对路径
        "workspace_dir": workspace,

        # 逻辑载荷：可以传入任何业务系统自带的结构化数据，供 Agent 或 Tool 随时调取
        "payload": {
            "user_id": "Theon"
        },

        "selected_skill": "",

        "error_counts": defaultdict(int),

        "plan": [],
        "past_steps": []
    }

    logger.info("开始流转状态机 (最大允许步数: 25) ...")
    run_config = {"recursion_limit": 25}

    skill_agent_app = construct_app()
    # 传入配置
    final_state = skill_agent_app.invoke(initial_state, run_config)

    logger.info("获取智能体输出结果")
    final_msg = final_state["messages"][-1]
    logger.info(f'\n\n最终输出为{final_msg.content}\n\n')



if __name__ == "__main__":
    user_query = '为我检查一下我上传的两张身份证照片是否是在同一个背景拍摄的，比如是否是同一张桌子？'
    main(user_query)
