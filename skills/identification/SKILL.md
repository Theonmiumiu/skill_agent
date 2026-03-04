---
name: id-background-verifier
description: 细致地验证两张证件照片的背景是否属于同一个真实物理位置。当需要防伪、比对证件拍摄现场一致性时触发。
---

# 技能概述
该技能解释如何进行身份证图片背景审核。主要流程是基于底层计算机视觉工具返回的量化数据，判定两张图片的背景是否为同一物理空间，并以标准的 JSON 格式输出结果。

# 技能
请按照以下步骤完成任务
1、调用 `verify_background_consistency` 工具对证件照背景一致性进行分析，获取 JSON 分析结果

# 可用工具
- `verify_background_consistency(image_path_1, image_path_2)`
  输入: 两个本地图像路径。
  输出 JSON:
    {
    "图片背景描述":"XXXXXXXXXX..."
    "判断依据":"XXXXXXX...",
    "审查结果":"通过"
    }

# 输出格式
照抄verify_background_consistency的结果即可

# 注意
严禁猜测或推断工具未返回的数据。
你的最终结论必须 100% 建立在 `verify_background_consistency` 工具返回的 JSON 字段上。
最终结果应该是 JSON 必须是合法的，不要在 JSON 外层包裹 ```json 的 Markdown 标记，确保业务系统可以直接解析。
