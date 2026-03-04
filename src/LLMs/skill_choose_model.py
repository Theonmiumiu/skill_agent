from .base_models import router_model, stream_wrapper
from pydantic import BaseModel, field_validator, ValidationError
from ..utils.logger import logger
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


template_prompt_system = """
    <role>
    你是一个严谨的技能挑选者
    </role>

    <task>
    你即将看到一组对话，你的工作是通过分析对话内容，定位出下一步应该使用什么技能，注意你只可以选一个技能。
    我们当前所有的技能和技能功能描述如下：
    {skills}
    </task>

    <output-format>
    你的输出应该是选定的工作流的"name"，注意严禁输出任何无关的内容，只输出单个工作流的"name"字段
    比如我们选定的skill的"name"字段为"test"，则直接输出"test"
    </output-format>
    """

template_prompt = ChatPromptTemplate(
    [
        ('system', template_prompt_system),
        MessagesPlaceholder('latest_context')
    ]
)

class SkillChooseModel:
    def __init__(self, registry):
        self.registry = registry
        self.parser = self.output_parser()

    def output_parser(self):
        # 为了验证输入的skill是否存在，得先把所有skill的名字提取出来
        name_set = set()
        for key in self.registry:
            name_set.add(key)

        class SkillName(BaseModel):
            # Literal只能写死，所以只能用Pydantic的field_validator
            name: str

            @field_validator('name')
            @classmethod
            def check_name(cls, v: str) -> str:
                if v not in name_set:
                    raise ValueError(f"无效的选择！必须是 {name_set} 之一，但收到了 '{v}'")
                return v

        return SkillName

    def invoke_prompt(self, latest_context: list):
        # 这里之后有拓展成RAG的空间
        skills_list = []
        for name in self.registry:
            skills_list.append(f'{name}：\n{self.registry[name]['description']}\n')
        skills = '\n'.join(skills_list)
        prompt = template_prompt.invoke(
            {
                'skills': skills,
                'latest_context': latest_context
            }
        )
        return prompt

    def choose(self, latest_context: list, max_retries: int = 2) -> str:
        """
        执行SKILL选择判别，包含 Pydantic 校验和重试机制。
        最大尝试次数 = 1(首次) + max_retries = 3 次。
        """
        prompt = self.invoke_prompt(latest_context)
        attempts = 0

        while attempts <= max_retries:
            raw_output = ''
            try:
                # 1. 调用云托管 LLM (网络波动与超时重试已交由 ChatOpenAI 底层接管)
                #llm_res = router_model.invoke(prompt)
                llm_res = stream_wrapper(router_model, prompt)
                # 安全提取字符串，并剔除大模型可能手贱加的前后空格或换行
                raw_output = llm_res.content.strip()
                # 3. 使用动态生成的 Pydantic 类进行严格校验
                # 注意：实例化 Pydantic 模型需要传入 keyword argument
                validated_data = self.parser(name=raw_output)
                # 如果代码能走到这里，说明校验完美通过
                logger.info(f"[choose_skills] SKILL选择: {validated_data.name}")
                return validated_data.name

            except ValidationError as e:
                # 捕获 Pydantic 校验失败 (格式不对、输出了废话等)
                attempts += 1
                logger.warning(
                    f"[choose_skills] 第 {attempts} 次格式校验失败。LLM 输出: '{raw_output}'。错误信息: {e}")

                if attempts > max_retries:
                    logger.error("[choose_skills] 达到最大重试次数，选择降级。")
                    # 容灾处理：如果重试全失败，返回空
                    return ""

            except Exception as e:
                # 2. 捕获 LLM API 鉴权失败、欠费等无法通过底层 max_retries 恢复的致命异常
                logger.error(f"[choose_skills] LLM 调用发生未知异常: {e}")
                return ""  # 触发降级
        return ""