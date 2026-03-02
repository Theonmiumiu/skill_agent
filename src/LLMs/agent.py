# agent.py
from .base_models import agent_model
from .basic_tools import *

class AgentModel:
    def __init__(self, registry, selected_skill):
        self.registry = registry
        self.selected_skill = selected_skill
        # 初始化时完成工具绑定
        self.model_with_tools = self.model_bind_tools()

    def model_bind_tools(self):
        # 四个系统工具一定要绑定
        basic_tools_list = [generate_skill,ask_user,choose_skills,check_workspace]
        # 防御性编程：如果没有命中 Skill，或者该 Skill 字典里没有配置 tools
        if not self.selected_skill or self.selected_skill not in self.registry:
            return agent_model.bind_tools(basic_tools_list)

        skill_tools_list = self.registry[self.selected_skill].get('tools', [])
        all_tools_list = skill_tools_list + basic_tools_list
        return agent_model.bind_tools(all_tools_list)


    def work(self, messages):
        # 传入 LangGraph 的 messages 列表进行调用
        res = self.model_with_tools.invoke(messages)
        return res

