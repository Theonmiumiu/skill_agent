from .base_models import router_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, field_validator, ValidationError
from ..utils.logger import logger

template_prompt_system = """
<role>
你是一个严谨的意图识别工具
</role>

<task>
你的工作是通过分析用户输入的需求，定位出之后应该触发什么工作流，注意你只可以选一个工作流。
我们当前所有的工作流和工作流功能描述如下：
{workflows}
</task>

<output-format>
你的输出应该是选定的工作流的"name"，注意严禁输出任何无关的内容，只输出单个工作流的"name"字段
</output-format>
"""

template_prompt = ChatPromptTemplate(
    [
        ('system', template_prompt_system),
        ('human', "{user_input}")
    ]
)


class RouterModel:
    def __init__(self, registry):
        self.registry = registry
        self.parser = self.output_parser()

    def output_parser(self):
        name_list = [key for key in self.registry]
        class SkillName(BaseModel):
        # Literal只能写死，所以只能用Pydantic的field_validator
            name: str
            @field_validator('name')
            @classmethod
            def check_name(cls, v: str) -> str:
                if v not in name_list:
                    raise ValueError(f"无效的选择！必须是 {name_list} 之一，但收到了 '{v}'")
                return v
        return SkillName

    def invoke_prompt(self, user_input):
        workflows_list = []
        for name in self.registry:
            workflows_list.append(f'{name}：\n{self.registry[name]['description']}\n')
        workflows = '\n'.join(workflows_list)
        prompt = template_prompt.invoke(
            {
                'workflows': workflows,
                'user_input': user_input
            }
        )
        return prompt

    def route(self, user_input: str, max_retries: int = 2) -> str:
        """
        执行路由判别，包含 Pydantic 校验和重试机制。
        最大尝试次数 = 1(首次) + max_retries = 3 次。
        """
        prompt = self.invoke_prompt(user_input)
        attempts = 0

        while attempts <= max_retries:
            raw_output = ''
            try:
                # 1. 调用云托管 LLM (网络波动与超时重试已交由 ChatOpenAI 底层接管)

                llm_res = router_model.invoke(prompt)

                # 安全提取字符串，并剔除大模型可能手贱加的前后空格或换行
                raw_output = llm_res.content.strip()

                # 3. 使用动态生成的 Pydantic 类进行严格校验
                # 注意：实例化 Pydantic 模型需要传入 keyword argument
                validated_data = self.parser(name=raw_output)

                # 如果代码能走到这里，说明校验完美通过
                logger.info(f"  [Router 成功] 意图识别命中: {validated_data.name}")
                return validated_data.name

            except ValidationError as e:
                # 捕获 Pydantic 校验失败 (格式不对、输出了废话等)
                attempts += 1
                logger.warning(f"  [Router 警告] 第 {attempts} 次格式校验失败。LLM 输出: '{raw_output}'。错误信息: {e}")

                if attempts > max_retries:
                    logger.error("  [Router 错误] 达到最大重试次数，路由降级。")
                    # 容灾处理：如果重试全失败，返回一个兜底的 Skill 名字，防止主程序崩溃
                    return "general_qa"

            except Exception as e:
                # 2. 捕获 LLM API 鉴权失败、欠费等无法通过底层 max_retries 恢复的致命异常
                logger.error(f"  [Router 致命错误] LLM 调用发生未知异常: {e}")
                return "general_qa"  # 触发降级
        return "general_qa"








