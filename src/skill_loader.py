# skill_loader.py
import frontmatter
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any
from langchain_core.tools import BaseTool
from .utils.logger import logger

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

