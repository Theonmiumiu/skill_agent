# processor.py
import operator
from typing import Annotated, Literal, TypedDict, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages
from .LLMs.basic_tools import generate_skill, ask_user, choose_skills, check_workspace
from .LLMs.base_models import actor_model, planner_model, reporter_model, stream_wrapper
from .utils.logger import logger
from collections import defaultdict
from .skill_loader import SkillLoader
from .LLMs.skill_choose_model import SkillChooseModel
import json5
import re

# ==========================================
# 2. 状态与节点定义 (LangGraph)
# ==========================================
# 初始化依赖对象，作用于全局，只在该脚本被main导入的使用用一次，之后会被更新后的覆盖
skill_manager = SkillLoader()
skill_choose_model = SkillChooseModel(skill_manager.registry)

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
    error_counts: defaultdict[str, int]

    # 引入草稿纸架构
    # 当前剩下的任务清单
    plan: List[str]
    # 历史已完成的步骤总结，使用 operator.add 意味着每次 return 都会 append 追加
    # 里面包含了每个actor调用的结果原文
    past_steps: Annotated[List[str], operator.add]


def skill_router_node(state: AgentState):
    """拦截 Agent 的技能切换请求，调用小模型分配技能"""
    logger.info("[Router] 接收到智能体的技能切换请求，正在重新分配...")
    last_msg = state["messages"][-1]

    messages_to_return = []
    new_skill = ""
    # 遍历大模型发出的所有 tool_call，给每一个都生成回复
    for tc in last_msg.tool_calls:
        if tc["name"] == "choose_skills":
            # 挑选技能的逻辑保持不变
            demand = HumanMessage(tc['args']['demand'])
            # 获取最近的有效上下文
            # 这个提示词没问题，只需要actor说一说将要处理的问题的描述，就能匹配上了
            context = [demand.content]
            new_skill = skill_choose_model.choose(context)
            if not new_skill:
                new_skill = ""
            messages_to_return.append(ToolMessage(
                content=f'系统已为你挂载技能: "{new_skill}"。请遵循新的技能和工具继续执行任务。',
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
        else:
            # 万一它同时调用了别的工具，直接打断并告诉它换技能了，之前的不算数
            messages_to_return.append(ToolMessage(
                content="由于触发了技能切换，该工具调用被系统取消。请根据新技能的SOP重新规划。",
                name=tc["name"],
                tool_call_id=tc["id"]
            ))

    return {
        "selected_skill": new_skill,
        "messages": messages_to_return,
        "plan": [],
        "past_steps": [f"【系统动作】智能体因遇到能力瓶颈，主动申请并成功切换了专属技能至: {new_skill}"]
    }

def dynamic_tools_node(state: AgentState):
    """【执行器】：根据选定的 Skill，动态拉取专属工具并执行"""
    logger.info("[Tools] 正在绑定Skill专属工具...")
    selected_skill = state.get("selected_skill")
    tool_error_counts = state['error_counts'].copy()
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
                        ToolMessage(content=f"工具执行异常: {str(e)}，请检查问题后尝试重新调用，不需要向上级或用户汇报此次错误", name=tool_name, tool_call_id=tool_call["id"])
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


def planner_node(state: AgentState):
    """负责宏观任务拆解与清单更新"""
    logger.info("[Planner] 正在审视全局，更新任务草稿纸...")

    # 获取用户的最最开始的任务
    # TODO事实上有点不灵活，不排除有人在一轮对话中突然大幅调整任务目标
    original_request = state["messages"][0].content if state["messages"] else "未知需求"

    # 获取当前的宏观上下文
    selected_skill = state.get("selected_skill", "")
    sop_content = skill_manager.registry.get(selected_skill, {}).get("sop_prompt", "无技能")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    # 取出最近的一条 agent 执行结果（如果是 agent 正常回复，说明上一步干完了）
    error_counts = state["error_counts"].copy()

    # 获取当前skill下的所有tools
    skill_tools_list = skill_manager.registry.get(selected_skill, {}).get('tools', [])
    basic_tools_list = [generate_skill, ask_user, choose_skills, check_workspace]
    all_tools_list = skill_tools_list + basic_tools_list
    # 格式化工具说明册
    tools_catalog = "\n".join([f"- 工具名: [{t.name}] | 功能: {t.description}" for t in all_tools_list])

    # 动态构建系统提示词，强制要求模型关注大局
    system_prompt = f"""
<role>
你是一个高效而果断任务规划主管，负责快速的规划拆接任务步骤，不进行具体执行。同时，你有一个下级执行你的任务列表中的任务。你们相互协作，因此不要亲自执行任务，只列出任务列表
</role>

<duty>
你的使命是基于用户的原始需求<original_user_request>，之前的计划<former_plan>以及已经完成的步骤<executed_task>，输出一份最新的、包含剩余的完成任务所需步骤计划的清单。并遵循以下约束：
1、参考技能<skill>中的说明编排任务，并输出编排好的任务列表，以便下级完成
2、你的输出应该是一个列表，列表中元素为任务的字符串，不要有任何除了这个列表之外的回答
3、如果你观察到<executed_task>已经完成了<skill>中的**除开按格式输出之外**的所有步骤，则固定输出["格式化输出"]
</duty>

<original_user_request>
这是用户最初始的核心诉求，你所有的计划拆解都必须为了服务于这个终极目标！
{original_request}
</original_user_request>

<workspace_dir>
当前工作目录 (Workspace): {state.get('workspace_dir', '未知')}
</workspace_dir>

<skill>
{sop_content}
请无视<skill>中的格式要求，你作为任务规划主管，总是输出列表！
</skill>

<tools>
这个tools不是给你用的，但你可以安排你的下级使用它们
{tools_catalog}
</tools>

<former_plan>
{plan if plan else '目前尚无计划，需要你从零生成。'}
</former_plan>

<executed_task>
{past_steps if past_steps else '目前刚开始，尚无已完成的步骤。'}
</executed_task>

<example_output>
[
  "调用 check_workspace 工具了解目录结构，寻找包含销售数据的文件",
  "调用 python_repl 工具读取数据文件，并计算总利润字段",
  "将计算得到的最终总利润数值汇报给用户"
]
</example_output>
"""

    # 组装消息，调用带有强制 JSON 输出约束的小模型
    notice = f'再次提醒你的任务是返回任务列表，而不是亲自执行任务！在下级已经完成**除开按格式输出之外**的所有步骤时，固定输出["格式化输出"]。\n你需要快速给出结果！快速给出结果！快速给出结果！得到可行结果后直接输出！严禁推敲和来回反思'
    messages = [SystemMessage(content=system_prompt), SystemMessage(content=notice)]  # 只给最近的对话防止污染

    response = stream_wrapper(planner_model, messages)
    match = re.search(r'\[.*?\]', response.content, re.DOTALL)
    if match:
        json_str = match.group(0)
        new_plan = json5.loads(json_str)
        logger.info(f"[Planner] 最新的任务清单已更新为: {new_plan}")
        error_counts["planner"] = 0
        return {"plan": new_plan, "error_counts": error_counts}
    elif error_counts['planner']<MAX_RETRY:
        logger.warning(f'[Planner] Planner节点似乎出现返回格式问题，尝试重新调用')
        error_counts["planner"] += 1
        return {"error_counts": error_counts}
    else:
        logger.warning(f'[Planner] Planner节点反复调用仍然失败，请检查模型原因')
        return {"plan":[], "past_steps":["planner模型内部失败，原因可能源于模型供应商，智能体系统崩溃"]}


def actor_node(state: AgentState):
    """行动的智能体：动态拼装标准上下文，执行工具调度"""
    plan = state.get("plan", [])
    if not plan:
        # 兜底逻辑，理论上不会走到这里，除非不需要计划
        current_task = "回应用户的直接提问或需求。"
    else:
        current_task = plan[0]
    logger.info(f"[Actor] 正在尝试完成步骤 {current_task} ...")

    # 提取用户的最初输入，让 Actor 也有个大局兜底（可选但推荐）
    original_user_msg = state["messages"][0] if state["messages"] else None
    # 【核心重构】：隔离出极其干净的“当前子任务沙盒”
    active_loop_messages = []

    # 我们倒序遍历消息列表
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and msg != original_user_msg:
            # 遇到用户中途的临时插话（比如 ask_user 的回答），保留并停止
            active_loop_messages.append(msg)
            break

        if isinstance(msg, AIMessage) and not msg.tool_calls:
            # 【极其关键】：遇到大模型输出的纯文本（说明上一个 plan[0] 刚刚汇报完工作了）
            # 这意味着上一个任务的生命周期结束，我们不需要再往前看了！
            break

        # 把属于当前任务循环的 ToolMessage 和 带有 tool_calls 的 AIMessage 收集起来
        if isinstance(msg, (AIMessage, ToolMessage)):
            active_loop_messages.append(msg)

    # 倒序回来，恢复正常的时间线
    active_loop_messages.reverse()
    # 不能知道用户的原始需求，否则要自作聪明

    system_prompt = f"""
<role>
你是一个专注而谨慎细心的执行者，负责使用手头的工具来解决任务，然后向你的上级汇报工作
</role>

<duty>
你的使命是使用技能完成任务，并严格遵循以下约束：
1、你拥有很多tool帮助你完成任务，请积极地使用它们，如果工具调用失败，可以依照工具的反馈进行重试或者放弃。你的回答里总应该包含工具输出的原文
2、当且仅当你确认该任务已经完全达成，或者彻底失败无法推进时，你**必须**输出一段总结性的文本（包含例如：“我收到的任务是什么“，”我已经完成了...操作，结果是...”，"工具调用的结果是..."）。
3、这段总结性文本将作为向上级汇报的内容，严禁输出任何与当前任务无关的事情，不要发问或者闲聊
</duty>

<task>
{current_task}
</task>

<past_steps_summary>
这是你之前已经完成的工作摘要，仅供参考，不要重复执行：
{state.get('past_steps', [])}
</past_steps_summary>
"""

    selected_skill = state.get("selected_skill")

    # 按着SKILL执行
    # 1. 获取该 Skill 的 SOP
    skill_tools_list = skill_manager.registry.get(selected_skill, {}).get('tools', [])
    basic_tools_list = [generate_skill, ask_user, choose_skills, check_workspace]
    all_tools_list = skill_tools_list + basic_tools_list
    # 2. 核心：将 workspace_dir 和 payload 作为系统级的环境变量，注入到 SOP 的末尾
    # 这样 Agent 既知道 SOP，又知道去哪里找文件，且完全不需要用户在提问中写明路径
    sop_context = (
        f"【系统环境变量】\n"
        f"- 当前工作目录 (Workspace): {state.get('workspace_dir', '未知')}\n"
        f"- 附加业务数据 (Payload): {state.get('payload', {})}"
    )
    system_msg = SystemMessage(system_prompt)
    sop_msg = HumanMessage(content=sop_context)

    messages_for_llm = [system_msg] + active_loop_messages + [sop_msg]

    # 5. 调用模型
    agent_with_tools = actor_model.bind_tools(all_tools_list)
    #response = agent_with_tools.invoke(messages_for_llm)
    response = stream_wrapper(agent_with_tools, messages_for_llm)

    # 【核心修复】：让干活的人自己去写总结汇报！
    updates = {"messages": [response]}

    # 如果 response 里没有 tool_calls，说明它不是在调工具，而是真的完成了这个子任务在输出总结！
    if not getattr(response, "tool_calls", None):
        finished_step_summary = f"当前任务是 '{current_task}'。执行结果汇报: {response.content}"
        updates["past_steps"] = [finished_step_summary]

    return updates

def reporter_node(state: AgentState):
    """【最终答复节点】：负责汇总所有成果，向用户做最终的优雅交付"""
    logger.info("[Reporter] 任务全部结束，正在撰写最终交付报告...")

    # 获取skill
    selected_skill = state.get("selected_skill", "")
    sop_content = skill_manager.registry.get(selected_skill, {}).get("sop_prompt", "无技能")

    # 1. 拿回用户的原始需求
    # TODO第一个输入可能没有不是当前需求，之后的改进方向
    original_request = state["messages"][0].content if state["messages"] else "未知需求"

    # 2. 拿出所有打工人踩过的坑和拿到的数据
    past_steps = state.get("past_steps", [])
    past_steps_text = "\n".join(past_steps) if past_steps else "未执行任何具体步骤。"

    system_prompt = f"""
    <role>
    你是一个冷酷、专业严谨的交付专员。你的后台团队已经为你收集了所有的必要信息。
    </role>

    <duty>
    你的唯一任务是：基于后台团队的【执行记录】，严格按照技能<skill>块中的输出格式要求输出结果
    如果后台团队的执行记录显示任务最终失败（例如没找到文件、权限不足），请仍然按照格式输出，但是内容解释为何任务失败
    </duty>

    <original_user_request>
    {original_request}
    </original_user_request>

    <backend_execution_records>
    {past_steps_text}
    </backend_execution_records>
    
    <skill>
    {sop_content}
    </skill>
    """
    # 只需要系统提示词就够了，直接生成最终回答
    # 我们调用那个带流式输出的 actor_model，并且同样可以使用 stream_wrapper 让它有思考过程

    messages_for_llm = [SystemMessage(content=system_prompt)]

    logger.info("[Reporter] 正在生成最终回答...")
    response_msg = stream_wrapper(reporter_model, messages_for_llm)

    # 追加到全局 messages 列表中，作为整个 Agent 运行的完美收尾
    return {"messages": [response_msg]}

def route_from_actor(state: AgentState) -> Literal["tools", "router", "planner"]:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        for tc in last_message.tool_calls:
            if tc["name"] == "choose_skills":
                return "router"

        return "tools"

    # 如果没调用工具，说明它干完了一步，输出了一段话，回去重新盘算草稿纸！
    return "planner"


def route_from_planner(state: AgentState) -> Literal["actor", "reporter", "planner"]:
    if state.get('error_counts')['planner'] != 0:
        return "planner"
    # 计划空了，说明大功告成，打完收工
    elif not state.get("plan", []) or state['plan'] == ['格式化输出']:
        return "reporter"

    # 还有计划，继续做
    return "actor"


# 别忘了在 skill_router_node 里加一行代码，清空旧计划：
# return {"selected_skill": new_skill, "messages": [success_msg], "plan": []}

def construct_app():
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("router", skill_router_node)
    workflow.add_node("actor", actor_node)
    workflow.add_node("tools", dynamic_tools_node)
    workflow.add_node("reporter", reporter_node)

    # 图的入口现在变成了 planner！上来先做计划！
    workflow.set_entry_point("planner")

    # 复杂的交通枢纽配置
    workflow.add_conditional_edges("planner", route_from_planner)
    workflow.add_conditional_edges("actor", route_from_actor)

    workflow.add_edge("tools", "actor")
    workflow.add_edge("router", "planner")  # 切完技能，强制回炉重造计划！
    workflow.add_edge("reporter", "__end__")

    return workflow.compile()

