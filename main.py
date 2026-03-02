# main.py
from collections import defaultdict
from src.processor import construct_app
from src.utils.logger import logger
from langchain_core.messages import HumanMessage
import json5


def main(query):
    # 构造完美符合 AgentState 结构的初始状态

    initial_state = {
        # 用户只需要说自然语言，不用管路径
        "messages": [HumanMessage(content=query)],

        # 物理边界：文件存在的绝对/相对路径
        "workspace_dir": "workspace",

        # 逻辑载荷：可以传入任何业务系统自带的结构化数据，供 Agent 或 Tool 随时调取
        "payload": {
            "user_id": "Theon"
        },

        "selected_skill": "",

        "tool_error_counts": defaultdict(int)
    }

    logger.info("开始流转状态机 (最大允许步数: 25) ...")
    run_config = {"recursion_limit": 25}

    try:
        skill_agent_app = construct_app()
        # 传入配置
        final_state = skill_agent_app.invoke(initial_state, run_config)

        logger.info("获取智能体输出结果")
        final_msg = final_state["messages"][-1]
        try:
            parsed_data = json5.loads(final_msg.content)
            logger.info(f'\n\n最终输出为{json5.dumps(parsed_data, indent=4, ensure_ascii=False)}\n\n')
        except Exception as e:
            logger.error(f'智能体输出结果json5解析失败：\n{e}')
            logger.info(f'智能体原始输出为{final_msg.content}')

    # 捕捉死循环异常，优雅退出而不是直接抛出红字报错
    except Exception as e:
        if "Recursion limit" in str(e):
            logger.error("触发死循环保护！Agent 尝试调用工具的次数过多，已强制终止。")
        else:
            logger.error(f"执行异常: {e}")
if __name__ == "__main__":
    user_query = '为我检查一下我上传的两张身份证照片是否是在同一个背景拍摄的，比如是否是同一张硕桌子？'
    main(user_query)
