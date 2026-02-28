# agent.py
from .base_models import agent_model


class AgentModel:
    def __init__(self, registry, selected_skill):
        self.registry = registry
        self.selected_skill = selected_skill
        # 初始化时完成工具绑定
        self.model_with_tools = self.model_bind_tools()

    def model_bind_tools(self):
        # 防御性编程：如果没有命中 Skill，或者该 Skill 字典里没有配置 tools
        if not self.selected_skill or self.selected_skill not in self.registry:
            return agent_model

        tool_list = self.registry[self.selected_skill].get('tools', [])

        if tool_list:
            # ✅ 正确做法：必须返回 bind_tools() 产生的新对象
            return agent_model.bind_tools(tool_list)

        return agent_model

    def work(self, messages):
        # 传入 LangGraph 的 messages 列表进行调用
        res = self.model_with_tools.invoke(messages)
        return res

