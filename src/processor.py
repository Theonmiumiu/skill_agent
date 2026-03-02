# processor.py
import frontmatter
import importlib.util
import inspect
from pathlib import Path
from typing import Annotated, Literal, TypedDict, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, add_messages
from .LLMs.agent import AgentModel
from .utils.logger import logger
from collections import defaultdict


# ==========================================
# 1. Skill 解析器 (核心：动态加载物理文件与工具)
# ==========================================
class SkillLoader:
    def __init__(self, base_dir: str = "skills"):
        self.base_dir = Path(base_dir)
        # 用于放注册在类中的skill
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.load_all_skills()

    def load_all_skills(self):
        """遍历物理文件夹，动态解析 SKILL.md 与 tools.py"""
        if not self.base_dir.exists():
            logger.error(f"[SkillLoader] 技能目录 {self.base_dir.resolve()} 不存在，请先创建！")
            return

        for skill_folder in self.base_dir.iterdir():
            if not skill_folder.is_dir():
                logger.warning(f'[SkillLoader] 在skills文件夹中发现非文件夹对象，请检查skills文件夹中的文件结构是否符合要求')
                continue
            # 各个skill文件夹里面必须至少要有SKILL.md
            md_path = skill_folder / "SKILL.md"
            # 各个可选文件夹
            scripts_path = skill_folder / 'scripts'
            tools_path = scripts_path / 'tools.py'
            # TODO下面这两个暂时都还没有涉及，没有做相应模块
            references_path = skill_folder / 'references'
            assets_path = skill_folder / 'assets'

            if md_path.is_file():
                # 1. 解析 YAML Frontmatter 和 SOP
                post = frontmatter.load(md_path)
                skill_name = post.metadata.get("name", skill_folder.name)
                skill_desc = post.metadata.get("description", "")
                sop_body = post.content.strip()

                # 2. 动态加载该 Skill 专属的 tools.py (如果存在)
                skill_tools = []
                if tools_path.is_file():
                    # IMPORTANT 当前的设计中所有的工具应该都在一个tools.py脚本下，之后也许可以优化
                    # 使用 importlib 动态执行外部 py 文件
                    # spec 对象包含了该文件的路径、加载器类型等信息。它还没读取文件内容，只是确认了“文件在哪儿”以及“怎么读”。
                    spec = importlib.util.spec_from_file_location(f"{skill_name}_tools", tools_path)
                    if spec and spec.loader:
                        # 根据刚才那份spec，在内存中创建一个全新的、空的 Python 模块对象。
                        module = importlib.util.module_from_spec(spec)
                        # 真正读取 tools.py 里的代码，并在刚才创建的 module 命名空间里执行这些代码
                        spec.loader.exec_module(module)

                        # 扫描模块中所有被 @tool 装饰的 LangChain 工具
                        for name, obj in inspect.getmembers(module):
                            if isinstance(obj, BaseTool):
                                skill_tools.append(obj)

                # 3. 注册到内存
                self.registry[skill_name] = {
                    "description": skill_desc,
                    "sop_prompt": sop_body,
                    "tools": skill_tools
                }

                tool_names = [t.name for t in skill_tools]
                logger.info(f"[SkillLoader] 注册Skill: {skill_name} | 挂载专属工具: {tool_names}")


# ==========================================
# 2. 状态与节点定义 (LangGraph)
# ==========================================
# 初始化依赖对象
skill_manager = SkillLoader()
MAX_RETRY = 2

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Agent有权限触碰的文件夹
    workspace_dir: str

    # 如果除了文件之外还有结构化的输入，可以从这里传入
    # 这是一个兜底的设计。如果前端不仅传了文件，还传了 {"客户层级": "VIP", "开户时间": "2026-02-27"}
    # TODO Agent应该具备在没获取所需的payload的时候调用工具主动向用户索要，在写工具的时候要有谱
    payload: Dict[str, Any]
    # 当前选择的SKILL
    selected_skill: str
    # 工具节点重试记录
    tool_error_counts: defaultdict[str, int]

def dynamic_tools_node(state: AgentState):
    """【执行器】：根据选定的 Skill，动态拉取专属工具并执行"""
    logger.info("[Tools] 正在绑定Skill专属工具...")

    selected_skill = state.get("selected_skill")
    tool_error_counts = state['tool_error_counts'].copy()
    # 动态获取当前 Skill 的专属工具
    current_skill_tools = skill_manager.registry.get(selected_skill, {}).get("tools", [])

    # 构建 O(1) 的查找字典
    tool_map = {t.name: t for t in current_skill_tools}

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
    return {"tool_error_counts":tool_error_counts, "messages": tool_outputs}

def llm_agent_node(state: AgentState):
    """【智能体大脑】：动态拼装标准上下文，执行工具调度"""
    logger.info("[Agent] 正在思考...")
    selected_skill = state.get("selected_skill")
    # TODO没有搭载SKILL的时候挑选SKILL分支
    #if not selected_skill:

    # TODO有SKILL的时候切换SKILL分支
    # TODO有SKILL的时候按着SKILL执行分支


    # 1. 获取该 Skill 的 SOP
    sop_content = skill_manager.registry.get(selected_skill, {}).get("sop_prompt", "你是一个得力的助手。")

    # 2. 核心：将 workspace_dir 和 payload 作为系统级的环境变量，注入到 SOP 的末尾
    # 这样 Agent 既知道 SOP，又知道去哪里找文件，且完全不需要用户在提问中写明路径
    sop_context = (
        f"【任务操作手册】\n"
        f"{sop_content}\n\n"
        f"【系统环境变量】\n"
        f"- 当前工作目录 (Workspace): {state.get('workspace_dir', '未知')}\n"
        f"- 附加业务数据 (Payload): {state.get('payload', {})}"
    )
    system_msg = HumanMessage(content=sop_context)

    # 3. 提取历史消息 (过滤掉可能混入的非标准 Message)，甚至可以排除之前的SKILL SOP的干扰，专注当下任务
    history = [m for m in state["messages"] if isinstance(m, (AIMessage, HumanMessage, ToolMessage))]

    # 4. 强制组装：System 永远在绝对的第一位！
    messages_for_llm = [system_msg] + history

    # 5. 调用模型
    # TODO这里似乎每次都要实例化，需要优化
    agent_model = AgentModel(skill_manager.registry, selected_skill)
    response = agent_model.work(messages_for_llm)

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

