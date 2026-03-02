# processor.py
from typing import Annotated, Literal, TypedDict, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages
from .LLMs.basic_tools import generate_skill, ask_user, choose_skills, check_workspace
from .LLMs.base_models import agent_model
from .utils.logger import logger
from collections import defaultdict
from .skill_loader import SkillLoader


# ==========================================
# 2. 状态与节点定义 (LangGraph)
# ==========================================
# 初始化依赖对象，作用于全局，只在该脚本被main导入的使用用一次，之后会被更新后的覆盖
skill_manager = SkillLoader()

# 超参数
MAX_RETRY = 2


class AgentState(TypedDict):
    # 上下文
    messages: Annotated[list, add_messages]
    # Agent有权限触碰的文件夹
    workspace_dir: str

    # 如果除了文件之外还有结构化的输入，可以从这里传入
    # 这是一个兜底的设计。如果前端不仅传了文件，还传了 {"客户层级": "VIP", "开户时间": "2026-02-27"}
    payload: Dict[str, Any]
    # 当前选择的SKILL
    selected_skill: str
    # 工具节点重试记录
    tool_error_counts: defaultdict[str, int]

def dynamic_tools_node(state: AgentState):
    """【执行器】：根据选定的 Skill，动态拉取专属工具并执行"""
    logger.info("[Tools] 正在绑定Skill专属工具...")
    messages = state.get('messages')
    selected_skill = state.get("selected_skill")
    tool_error_counts = state['tool_error_counts'].copy()
    # 动态获取当前 Skill 的专属工具
    current_skill_tools = skill_manager.registry.get(selected_skill, {}).get("tools", [])

    # 构建 O(1) 的查找字典
    # 这里得加上智能体的四个通用技能避免找不到
    tool_map = {t.name: t for t in current_skill_tools}
    basic_tools_map = {"generate_skill":generate_skill,"ask_user":ask_user,"choose_skills":choose_skills,"check_workspace":check_workspace}
    tool_map.update(basic_tools_map)

    last_msg = state["messages"][-1]
    tool_outputs = []

    # 遍历大模型请求的所有工具调用
    # 大模型的工具请求Message对象会有tool_calls属性
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f'[Tools] 智能体请求执行工具 {tool_name} ')

        if tool_name in tool_map:
            # 找到对应工具并执行
            tool_instance = tool_map[tool_name]
            try:
                # IMPORTANT 除开几个basic_tool允许有注入参数，因为需要管理上下文和技能加载器之外，其他tool不应该有注入参数
                if tool_name == 'choose_skills':
                    # 直接覆盖就行，反正这个tool，LLM不会传参
                    tool_args = {
                        "latest_context": messages[-10:],
                        "skill_manager": skill_manager
                    }
                    # 获取选择加载的skill名称，搭载skill
                    selected_skill = tool_instance.invoke(tool_args)
                    tool_outputs.append(
                        ToolMessage(content=f'[技能选取] 成功选取技能 "{selected_skill}"', name=tool_name, tool_call_id=tool_call["id"])
                    )

                else:
                    result = tool_instance.invoke(tool_args)
                    tool_outputs.append(
                        ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_call["id"])
                    )
                logger.info(f'[Tools] 工具 {tool_name} 成功执行')
                # 成功调用，重试计数清零
                tool_error_counts[tool_name] = 0
            except Exception as e:
                retry = tool_error_counts[tool_name]
                if retry < MAX_RETRY:
                    logger.warning(f'[Tools] 工具 {tool_name} 执行异常，执行第 {retry+1} 次重试')
                    tool_outputs.append(
                        ToolMessage(content=f"工具执行异常: {str(e)}，请检查问题后尝试重新调用", name=tool_name, tool_call_id=tool_call["id"])
                    )
                    tool_error_counts[tool_name] +=1
                else:
                    logger.error(f'[Tools] 工具 {tool_name} 执行错误！执行 {MAX_RETRY} 次重试依然失败，要求智能体放弃该工具')
                    tool_outputs.append(
                        ToolMessage(content=f"工具连续出错！请放弃调用该工具！放弃调用该工具！基于之前所得所有信息执行下一步！", name=tool_name,
                                    tool_call_id=tool_call["id"])
                    )
        else:
            # 安全兜底：如果模型幻觉调用了不在当前 Skill 里的工具，直接拦截并报错回传
            retry_exception = tool_error_counts['exception']
            if retry_exception < MAX_RETRY:
                error_msg = f"非法调用：当前技能 [{selected_skill}] 中不存在工具 '{tool_name}'，请检查问题后尝试重新调用。"
                logger.error(f'[Tools] {error_msg}')
                tool_outputs.append(
                    ToolMessage(content=error_msg, name=tool_name, tool_call_id=tool_call["id"])
                )
                tool_error_counts['exception'] +=1
            else:
                logger.error(f'[Tools] 工具节点执行错误！执行 {MAX_RETRY} 次重试依然调用了不存在的工具，要求智能体放弃该工具')
                tool_outputs.append(
                    ToolMessage(content=f"工具连续出错！请放弃调用该工具！放弃调用该工具！基于之前所得所有信息执行下一步！",
                                name=tool_name,
                                tool_call_id=tool_call["id"])
                )

    # 增量更新状态
    return {"tool_error_counts":tool_error_counts, "messages": tool_outputs, "selected_skill": selected_skill}

def llm_agent_node(state: AgentState):
    """【智能体大脑】：动态拼装标准上下文，执行工具调度"""
    logger.info("[Agent] 正在思考...")
    system_prompt = """
<role>
你是一个专注而谨慎细心的智能体，负责解决用户提出的各类复杂任务。
</role>

<duty>
你的使命是使用技能完成用户指派的各类任务，并严格遵循以下约束：
1、对于不清楚的任务细节，总是主动询问用户确认，确保你的理解正确
2、对于用户没有提供的任务所需参数和材料，总是主动询问用户索要
3、完成任务前总是挑选技能，然后通过技能的标准流程完成任务
4、你拥有很多tool帮助你完成任务，请积极地使用它们
5、开始完成任务前总是与用户确认，获得用户认可之后再执行计划，计划执行中禁止再询问用户，因此总是在执行任务前确认好细节问题
</duty>
        """

    selected_skill = state.get("selected_skill")

    # 按着SKILL执行
    # 1. 获取该 Skill 的 SOP
    sop_content = skill_manager.registry.get(selected_skill, {}).get("sop_prompt", "你是一个得力的助手。")
    skill_tools_list = skill_manager.registry.get(selected_skill, {}).get('tools', [])
    basic_tools_list = [generate_skill, ask_user, choose_skills, check_workspace]
    all_tools_list = skill_tools_list + basic_tools_list
    # 2. 核心：将 workspace_dir 和 payload 作为系统级的环境变量，注入到 SOP 的末尾
    # 这样 Agent 既知道 SOP，又知道去哪里找文件，且完全不需要用户在提问中写明路径
    sop_context = (
        f"【任务操作手册】\n"
        f"{sop_content}\n\n"
        f"【系统环境变量】\n"
        f"- 当前工作目录 (Workspace): {state.get('workspace_dir', '未知')}\n"
        f"- 附加业务数据 (Payload): {state.get('payload', {})}"
    )
    system_msg = SystemMessage(system_prompt)
    sop_msg = HumanMessage(content=sop_context)

    # 3. 提取历史消息 (过滤掉可能混入的非标准 Message)，甚至可以排除之前的SKILL SOP的干扰，专注当下任务
    history = [m for m in state["messages"] if isinstance(m, (AIMessage, HumanMessage, ToolMessage))]
    messages_for_llm = [system_msg] + history + [sop_msg]

    # 5. 调用模型
    agent_with_tools = agent_model.bind_tools(all_tools_list)
    response = agent_with_tools.invoke(messages_for_llm)

    # 6. 将模型的输出增量更新回全局状态
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


# 启动时执行挂载
def construct_app():
    # ==========================================
    # 3. 构建流转图 (ReAct State Machine)
    # 似乎是个非常简单的图，甚至可能过于简单了
    # ==========================================
    # 构建流转图
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", llm_agent_node)
    # 直接挂载我们手写的动态节点，抛弃 prebuilt.ToolNode
    workflow.add_node("tools", dynamic_tools_node)

    workflow.set_entry_point("agent")
    # 这个should_continue是个函数，会返回结束或者tools的字符串，也即节点名
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent") # 工具执行完切回 Agent

    skill_agent_app = workflow.compile()
    return skill_agent_app

