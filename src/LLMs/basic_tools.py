# basic_tools.py
from langchain_core.tools import tool
from typing import Annotated
import seedir as sd
from ..utils.logger import logger

@tool()
def check_workspace(
        workspace: Annotated[str, "任务所需的材料存放的文件夹绝对路径，例如 '/app/workspace'"]
) -> str:
    """
    用于穿透workspace文件夹中的文件结构，了解workspace文件夹中有什么内容，以便准确获取任务所需材料
    :return: 返回文件夹的层次结构以及各文件名称
    """
    tree_str = sd.seedir(workspace,printout=False)
    return tree_str

# 上下文的Message列表结构是不暴露给LLM的，所以不能作为参数传进去，得在主流程里面管理
@tool()
def choose_skills(
    demand: Annotated[str, "你现在需要解决什么问题，想要什么功能的工具"]
) -> str:
    """
    用于选取技能的函数，会为你分配可以用于完成当前任务的技能，从而能依据技能的流程完成任务。
    调用此函数时严禁同时调用其他函数！
    总是以下情况调用该工具：
    1、你当前的技能无法完成下一步任务时
    函数将为你挂载所需的技能
    :return: 完成任务的所需技能
    """
    # 其实返回值没用
    return "请求调用技能路由"

@tool()
def ask_user(
        query: Annotated[str, "你向用户询问的问题或者寻求确认的问题，应该总是分点询问"]
) -> str:
    """
    当你对当前任务的有问题时，严禁自己猜测！总是调用该函数向用户确认。
    在如下情况总是调用该函数：
    1、用户对任务的描述存在歧义时
    2、用户没有提供完整的任务参数时
    3、用户的任务描述与workspace中的任务资料不吻合时
    注意，如果用户的回答仍然不能解答你的问题，可以频繁连续调用该函数，以对任务有一个清晰明确的认知
    :return: 返回用户的回答
    """
    # TODO这个地方之后需要使用langgraph interrupt机制才行，现在也就单机demo不会有问题罢了
    front_prompt = f"""

------------------------------------
[智能体] 希望与您对齐任务需求：
{query}

------------------------------------
请给他一些指示：
"""
    user_res = input(front_prompt)
    return user_res


# TODO这个模块太大了，先把SKILL应用模块开发完再来动这个SKILL开发模块
@tool()
def generate_skill():
    """
    工具暂时无效，禁止使用该工具
    :return:
    """
    pass

