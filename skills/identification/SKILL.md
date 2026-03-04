---
name: id-background-verifier
description: 细致地验证两张证件照片的背景是否属于同一个真实物理位置。当需要防伪、比对拍摄现场一致性时触发。
---

# 技能概述
该技能解释如何进行身份证图片背景审核。主要流程是基于底层计算机视觉工具返回的量化数据，判定两张图片的背景是否为同一物理空间，并以标准的 JSON 格式输出结果。

# 技能
请按照以下步骤完成任务
1、调用 `verify_background_consistency` 工具对证件照背景一致性进行分析，获取 JSON 分析结果
2、根据分析结果进行总结，按照格式输出，格式如下：
{
  "审核结论": "PASS | REJECT | REVIEW",
  "核心数据": {
    "特征点数": [填入整数],
    "置信度": [填入浮点数],
    "空间透视状态": "有效 | 无效"
  },
  "判定解释": "[一句话解释判定原因，例如：虽然背景颜色相似，但几何特征点完全不匹配，置信度仅为 0.12。]",
  "工具输出": {
    "is_same_place": [原样填入工具输出的布尔值],
    "inliers_count": [原样填入工具输出的整数],
    "homography_valid": [原样填入工具输出的布尔值],
    "confidence_score": [原样填入工具输出的浮点数]
  }
}

# 可用工具
- `verify_background_consistency(image_path_1, image_path_2)`
  输入: 两个本地图像路径。
  输出 JSON:
  - `is_same_place` (bool): 算法判定结论
  - `inliers_count` (int): 空间几何匹配的有效特征点数
  - `homography_valid` (bool): 空间透视关系是否成立
  - `confidence_score` (float): 判定置信度 (0.0-1.0)


# 判断依据
必须严格按照以下阈值映射判定结果：
1. [PASS]
   条件: `is_same_place` == true AND `inliers_count` >= 15 AND `confidence_score` >= 0.85
   结论: 背景空间透视关系完全一致，属于同一物理地点。
2. [REJECT]
   条件: `is_same_place` == false AND `inliers_count` < 5
   结论: 物理特征断裂或空间关系不匹配，属于不同地点。
3. [REVIEW]
   条件: 不满足上述两类的中间态（例如特征点数量极低，如纯白/纯黑背景）。
   结论: 缺乏足够防伪纹理特征，系统判定置信度不足，拦截转人工。

# 注意
严禁猜测或推断工具未返回的数据。
你的最终结论必须 100% 建立在 `verify_background_consistency` 工具返回的 JSON 字段上。
最终结果应该是 JSON 必须是合法的，不要在 JSON 外层包裹 ```json 的 Markdown 标记，确保业务系统可以直接解析。
