import json5
import frontmatter
import importlib.util
import inspect
from pathlib import Path
from typing import Annotated, Literal, TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from .LLMs.router import RouterModel


# ==========================================
# 1. Skill 解析器 (核心：动态加载物理文件与工具)
# ==========================================
class SkillLoader:
    def __init__(self, base_dir: str = "skills"):
        self.base_dir = Path(base_dir)
        # 用于放注册在类中的skill
        self.registry: Dict[str, Dict[str, Any]] = {}
        # 汇总所有加载到的工具
        self.all_tools: List[BaseTool] = []
        self.load_all_skills()

    def load_all_skills(self):
        """遍历物理文件夹，动态解析 SKILL.md 与 tools.py"""
        if not self.base_dir.exists():
            print(f"[警告] 技能目录 {self.base_dir.resolve()} 不存在，请先创建！")
            return

        for skill_folder in self.base_dir.iterdir():
            if not skill_folder.is_dir():
                continue
            # 各个skill文件夹里面必须至少要有SKILL.md，
            md_path = skill_folder / "SKILL.md"
            tools_path = skill_folder / "tools.py"

            if md_path.is_file():
                # 1. 解析 YAML Frontmatter 和 SOP
                post = frontmatter.load(md_path)
                skill_name = post.metadata.get("name", skill_folder.name)
                skill_desc = post.metadata.get("description", "")
                sop_body = post.content.strip()

                # 2. 动态加载该 Skill 专属的 tools.py (如果存在)
                skill_tools = []
                if tools_path.is_file():
                    # 使用 importlib 动态执行外部 py 文件
                    spec = importlib.util.spec_from_file_location(f"{skill_name}_tools", tools_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # 扫描模块中所有被 @tool 装饰的 LangChain 工具
                        for name, obj in inspect.getmembers(module):
                            if isinstance(obj, BaseTool):
                                skill_tools.append(obj)
                                self.all_tools.append(obj)

                # 3. 注册到内存
                self.registry[skill_name] = {
                    "description": skill_desc,
                    "sop_prompt": sop_body,
                    "tools": skill_tools
                }

                tool_names = [t.name for t in skill_tools]
                print(f"[加载完毕] Skill: {skill_name} | 挂载专属工具: {tool_names}")


# ==========================================
# 2. 状态与节点定义 (LangGraph)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    # 1. 物理边界：工作区目录 (Workspace)
    # 前端只需在服务器上建一个临时文件夹（UUID命名），把用户上传的 1个、2个 甚至 10个文件全扔进去。
    # Agent 只认这一个文件夹路径。
    workspace_dir: str

    # 2. 逻辑载荷：通用数据字典 (Payload)
    # 这是一个兜底的设计。如果前端不仅传了文件，还传了 {"客户层级": "VIP", "开户时间": "2026-02-27"}
    # 统统装进这个字典里，引擎本身不需要知道里面有什么，按需提取。
    payload: Dict[str, Any]

    # 3. 系统控制流
    selected_skill: str

def router_node(state: AgentState):
    """【意图识别】：仅依据 Description 路由"""
    print("\n[节点: Router] 正在通过 Skill Description 识别意图...")
    user_query = state["messages"][0].content
    chosen_skill = router_model.route(user_query)
    print(f"-> 命中 Skill: [{chosen_skill}]")

    # 注入该 Skill 的 SOP
    sop_content = skill_manager.registry.get(chosen_skill, {}).get("sop_prompt", "你是一个得力的助手。")
    system_msg = SystemMessage(content=sop_content)

    return {"messages": [system_msg], "selected_skill": chosen_skill}


def llm_agent_node(state: AgentState):
    """【智能体大脑】：GLM-5 思考逻辑"""
    print("\n[节点: LLM Agent] GLM-5 正在基于 SOP 思考...")

    history = [m for m in state["messages"] if isinstance(m, (AIMessage, HumanMessage)) or m.type == "tool"]
    last_msg = history[-1] if history else None

    # TODO======= [模拟 LLM 工具调用逻辑] =======
    if isinstance(last_msg, HumanMessage):
        print("-> LLM 决定调用工具：extract_text_from_file")
        mock_ai_msg = AIMessage(content="", tool_calls=[
            {"name": "extract_text_from_file", "args": {"file_path": state["file_path"]}, "id": "call_1"}])
        return {"messages": [mock_ai_msg]}
    elif getattr(last_msg, 'name', '') == 'extract_text_from_file':
        print("-> LLM 决定并行调用工具：check_fee_rate & check_risk_warning")
        mock_ai_msg = AIMessage(content="", tool_calls=[
            {"name": "check_fee_rate", "args": {"text": last_msg.content}, "id": "call_2"},
            {"name": "check_risk_warning", "args": {"text": last_msg.content}, "id": "call_3"}
        ])
        return {"messages": [mock_ai_msg]}
    else:
        print("-> LLM 收集齐结果，生成最终 JSON5 报告")
        messy_json5_from_llm = """{
            // 这是 GLM-5 的输出结果
            'status': "违规", 
            risk_level: '高风险',
            reason: "费率过低且缺失风险揭示语",
        }"""
        return {"messages": [AIMessage(content=messy_json5_from_llm)]}
    # ========================================


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "__end__"


# 启动时执行挂载
skill_manager = SkillLoader()
router_model = RouterModel(skill_manager.registry)


# ==========================================
# 3. 构建流转图 (ReAct State Machine)
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("agent", llm_agent_node)

# 动态绑定所有扫描到的物理工具
if skill_manager.all_tools:
    tool_node = ToolNode(skill_manager.all_tools)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("tools", "agent")
else:
    # 防止因完全没有配置工具导致图编译失败的兜底
    def dummy_tools(state):
        return {"messages": []}


    workflow.add_node("tools", dummy_tools)
    workflow.add_edge("tools", "agent")

workflow.set_entry_point("router")
workflow.add_edge("router", "agent")
workflow.add_conditional_edges("agent", should_continue)

skill_agent_app = workflow.compile()

# ==========================================
# 4. 执行测试
# ==========================================
if __name__ == "__main__":
    print(">>> 启动引擎...\n")

    initial_state = {
        "messages": [HumanMessage(content="帮我审查一下这两张身份证照片是不是同一个地方拍的。")],
        "file_path": "/uploads/promo_poster_01.pdf",
        "selected_skill": ""
    }

    # 如果找不到真实目录，引擎会安全地空转
    if not skill_manager.registry:
        print("请按照结构在本地建立 skills 文件夹后再运行测试！")
    else:
        final_state = skill_agent_app.invoke(initial_state)

        print("\n========== [解析后] JSON 报告 ==========")
        try:
            parsed_data = json5.loads(final_state["messages"][-1].content)
            print(json5.dumps(parsed_data, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"解析失败: {e}")