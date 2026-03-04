# tools.py
import cv2
import numpy as np
import json
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from typing import Annotated
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
import re
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
import json5

class Uploader:
    """
    负责链接云数据库并负责上传的实例
    """
    def __init__(self):
        self.ak = os.getenv("TOS_AK")
        self.sk = os.getenv("TOS_SK")
        self.endpoint = os.getenv("OSS_ENDPOINT")
        self.bucket_name = os.getenv("TOS_BUCKET_NAME")
        self.allowed_extensions = {
        # === 文档类 ===
        '.pdf',                 # PDF
        '.doc', '.docx',        # Word
        '.txt', '.rtf',         # 纯文本/富文本

        # === 图片类 ===
        '.jpg', '.jpeg',        # JPEG
        '.png',                 # PNG
        '.gif',                 # GIF
        '.bmp',                 # Bitmap
        '.webp',                # WebP (现代网页常用)
        '.tiff', '.tif',        # TIFF (印刷/扫描常用)
        '.ico',                 # 图标

        # === 音频类 ===
        '.mp3',                 # MP3
        '.wav',                 # WAV (无损)
        '.aac', '.m4a',         # AAC/M4A (苹果设备常用)
        '.flac',                # FLAC (发烧友无损)
        '.ogg',                 # OGG
        '.wma',                 # WMA
        '.amr',                 # AMR (老式录音/语音)
        '.opus',                # OPUS (高效语音编码，WhatsApp/Telegram常用)

        # === 视频类 ===
        '.mp4',                 # MP4 (最通用)
        '.mov',                 # MOV (QuickTime)
        '.avi',                 # AVI
        '.mkv',                 # MKV (虽然兼容性一般，但很多高清资源是这个)
        '.webm',                # WebM (网页视频)
        '.flv',                 # FLV (老式Flash视频)
        '.wmv',                 # WMV
        '.mpeg', '.mpg',        # MPEG
        '.m4v',                 # M4V
        '.3gp',                 # 3GP (老手机视频)
        '.ts',                  # TS流
    }

        tos_config = Config(
            region_name='cn-shanghai',
            s3={'addressing_style': 'virtual'}
        )

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.ak,
            aws_secret_access_key=self.sk,
            endpoint_url=self.endpoint,
            config=tos_config
        )

    def _check_file_exists(self, object_name: str)->bool:
        """
        检查文件在云端是否存在
        :param object_name: 在云端上的文件名
        :return: 返回是否存在的判断
        """
        try:
            # head_object 只获取文件元数据，不下载内容，速度极快且不怎么耗流量
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError:
            # 如果报 404 Not Found，说明文件不存在
            return False

    def upload_and_get_presigned_url(self, file_path: Path, file_type: str, expiration=3600):
        """
        将本地文件上传到云端，获取url便于传给模型
        :param file_path: 文件在本地的路径
        :param file_type: 文件的类型，暂时可以接受text,image,video,audio,file，作用只是把不同的文件归到不同的文件夹下
        :param expiration: 云端链接有效期
        :return: 云端容器上文件的url
        """

        def _sanitize_filename(name):
            """
            辅助函数，用于得到安全文件名
            """
            # 1. 替换非法字符为下划线
            # \ / : * ? " < > | 以及 控制字符
            name = re.sub(r'[\/\\:\*\?"<>| \x00-\x1f]', '_', name)

            # 2. 去除首尾的空格和句点
            name = name.strip(". ")

            # 3. 检查 Windows 保留字
            reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3",
                              "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                              "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6",
                              "LPT7", "LPT8", "LPT9"}
            if name.upper() in reserved_names:
                name = f"_{name}"

            # 4. 截断长度（防止过长）
            return name[:250]


        try:
            ext = file_path.suffix
            if ext not in self.allowed_extensions:
                print(f'上传的文件的文件类型不受支持：{ext}')
                return None
            stem = _sanitize_filename(file_path.stem)
            if file_type in {'text','image','video','audio','file'}:
                object_name = f"{file_type}/{stem}{ext}"
            else:
                print(f'传入的文件类型不受支持：{file_type}')
                return None

            #【缓存命中检查】检查云端是否已经有这个文件
            if self._check_file_exists(object_name):
                print(f"⚡ [缓存命中] 文件已存在，跳过上传: {object_name}")
            else:
                # 缓存未命中，执行上传
                print(f"⬆️ [新文件] 正在上传: {object_name}")
                self.s3_client.upload_file(str(file_path), self.bucket_name, object_name)

            # 无论是否新上传，都生成预签名 URL (生成 URL 是本地计算，无需网络请求)
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url

        except NoCredentialsError:
            print("❌ 凭证错误，请检查 .env 配置")
            return None
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            return None


def _mask_id_card(image: np.ndarray) -> np.ndarray:
    """
    精细版辅助函数：利用几何特征和宽高比，动态识别并遮挡身份证区域。
    （已新增：自动将中间结果保存到本地 debug_masks_output 文件夹）
    """
    h, w = image.shape[:2]
    # 初始化全白（255）的掩膜
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # 1. 预处理：降噪（如果传入的已经是灰度图，直接使用）
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 边缘检测 (Canny)
    edges = cv2.Canny(blur, 50, 150)

    # 3. 闭运算 (Morphology Close)：连接断裂的边缘线条
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 4. 寻找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积降序排序，只看画面中最大的前 5 个轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    id_contour = None
    for c in contours:
        # 计算轮廓周长
        peri = cv2.arcLength(c, True)
        # 多边形拟合，逼近真实的几何形状
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 寻找拥有 4 个顶点（矩形）的轮廓
        if len(approx) == 4:
            area = cv2.contourArea(c)
            # 过滤掉太小的噪点（假设身份证至少占整图的 10% 面积）
            if area > (h * w * 0.1):
                # 获取正交边界框，计算宽高比
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect_ratio = max(bw, bh) / float(min(bw, bh))

                # 校验宽高比是否符合标准证件特征 (放宽容差区间在 1.3 到 1.8 之间)
                if 1.3 < aspect_ratio < 1.8:
                    id_contour = approx
                    break

    # 5. 绘制并应用掩膜
    if id_contour is not None:
        # 找到了身份证！将其涂黑 (0)
        cv2.drawContours(mask, [id_contour], -1, 0, -1)

        # 安全操作：稍微向外膨胀一点黑色区域（腐蚀白色掩膜）
        kernel_erode = np.ones((20, 20), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=1)
    else:
        # 【降级方案 (Fallback)】
        x1, y1 = int(w * 0.25), int(h * 0.25)
        x2, y2 = int(w * 0.75), int(h * 0.75)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    # ==========================================
    # 新增：自动保存验证图片到本地
    # ==========================================
    try:
        # 在运行代码的同级目录下创建一个 debug 文件夹
        debug_dir = "debug_masks_output"
        os.makedirs(debug_dir, exist_ok=True)

        # 获取当前时间戳，防止多次运行覆盖前面的图片
        timestamp = int(time.time() * 1000)

        # 为了直观展示，如果原图是单通道灰度图，先临时转回三通道
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()

        # 核心：将 mask 盖回原图。这里会把原图里的身份证位置变成纯黑，木桌背景保留原样
        visualized_result = cv2.bitwise_and(vis_image, vis_image, mask=mask)

        # 1. 保存纯黑白的算法底层 Mask (0和255)
        cv2.imwrite(os.path.join(debug_dir, f"1_mask_logic_{timestamp}.jpg"), mask)
        # 2. 保存你最关心的“涂黑后的原图”
        cv2.imwrite(os.path.join(debug_dir, f"2_mask_applied_{timestamp}.jpg"), visualized_result)

        # 静默执行，不打印过多日志干扰控制台
    except Exception as e:
        print(f"  [Debug] 图片保存失败: {str(e)}")
    # ==========================================

    return mask


def cv_method(
        image_path_1: Annotated[str, "第一张需要分析的照片的文件路径"],
        image_path_2: Annotated[str, "第二张需要分析的照片的文件路径"]
) -> str:
    """
    使用统计学视觉特征评估两张身份证照片的背景一致性。这个工具经过优化，足够全面和稳健
    输出通过cv算法得出的"颜色相关性"、"纹理相关性"、"亮度相似性"，越接近1表明越相关或者相似，
    """
    # 1. 加载图像并校验
    img1 = cv2.imread(image_path_1)
    img2 = cv2.imread(image_path_2)

    if img1 is None or img2 is None:
        return {"error": "无法读取图片，请检查路径。"}

    # 2. 提取并合并背景掩膜
    # 生成各自的掩膜 (背景=255, 证件=0)
    mask1 = _mask_id_card(img1)
    mask2 = _mask_id_card(img2)

    # 关键步骤：取两个掩膜的交集 (bitwise_and)
    # 这样确保我们只比较两张照片中 *共同暴露* 的背景区域，不受证件位置变化的影响
    common_mask = cv2.bitwise_and(mask1, mask2)

    # 如果公共背景区域太小（比如两张身份证刚好占据了完全不同的屏幕角落），直接返回低分
    if cv2.countNonZero(common_mask) < (img1.shape[0] * img1.shape[1] * 0.1):
         return {"error": "公共背景区域过小，无法有效提取特征。"}

    # ==========================================
    # 维度一：全局颜色分布相似度 (HSV Color Histogram)
    # ==========================================
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # 提取 Hue(色相) 和 Saturation(饱和度) 通道的 2D 直方图
    hist_color1 = cv2.calcHist([hsv1], [0, 1], common_mask, [50, 60], [0, 180, 0, 256])
    hist_color2 = cv2.calcHist([hsv2], [0, 1], common_mask, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_color1, hist_color1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_color2, hist_color2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 使用相关性(Correlation)比较，范围 [-1, 1]
    color_score = cv2.compareHist(hist_color1, hist_color2, cv2.HISTCMP_CORREL)

    # ==========================================
    # 维度二：全局纹理结构相似度 (Sobel Gradient Histogram)
    # ==========================================
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用 Sobel 算子计算图像的局部梯度（捕捉木纹、缝隙等纹理特征）
    grad_x1, grad_y1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0), cv2.Sobel(gray1, cv2.CV_32F, 0, 1)
    grad_x2, grad_y2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0), cv2.Sobel(gray2, cv2.CV_32F, 0, 1)

    mag1 = cv2.magnitude(grad_x1, grad_y1)
    mag2 = cv2.magnitude(grad_x2, grad_y2)

    # 映射回 0-255 并计算直方图
    mag1_8u = np.uint8(np.clip(mag1, 0, 255))
    mag2_8u = np.uint8(np.clip(mag2, 0, 255))

    hist_tex1 = cv2.calcHist([mag1_8u], [0], common_mask, [64], [0, 256])
    hist_tex2 = cv2.calcHist([mag2_8u], [0], common_mask, [64], [0, 256])

    cv2.normalize(hist_tex1, hist_tex1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_tex2, hist_tex2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    texture_score = cv2.compareHist(hist_tex1, hist_tex2, cv2.HISTCMP_CORREL)

    # ==========================================
    # 维度三：环境光照一致性 (Mean Brightness Check)
    # ==========================================
    mean_val1 = cv2.mean(gray1, mask=common_mask)[0]
    mean_val2 = cv2.mean(gray2, mask=common_mask)[0]

    # 将绝对差异转化为 0 到 1 的得分（假设最大差异阈值为 255）
    brightness_diff = abs(mean_val1 - mean_val2)
    brightness_score = max(0.0, 1.0 - (brightness_diff / 255.0))

    # ==========================================
    # 综合判定
    # ==========================================
    # 将负相关修正为 0
    color_score = max(0.0, float(color_score))
    texture_score = max(0.0, float(texture_score))

    result = {
        "color_correlation": round(color_score, 3),
        "texture_correlation": round(texture_score, 3),
        "brightness_score": round(brightness_score, 3)
    }

    return json.dumps(result)

# 获取环境变量
current_dir = Path(__file__).parent
env_path = current_dir / ".env"

load_dotenv(env_path, override=True)
model = os.getenv("MODEL")
llm_url = os.getenv("LLM_URL")
llm_api_key = os.getenv("LLM_API_KEY")
# 相似度接收阈值超参数
THRESHOLD = 0.75

# 初始化上传器实例
uploader = Uploader()

visual_llm = ChatOpenAI(
    model = model,
    base_url = llm_url,
    api_key = llm_api_key,
    temperature = 0.1,
    top_p=0.5,
    max_retries=2,
    # extra_body={
    #     "enable_thinking" : False
    # }
)

@tool()
def verify_background_consistency(
        image_path_1: Annotated[str, "第一张需要分析的照片的文件路径"],
        image_path_2: Annotated[str, "第二张需要分析的照片的文件路径"]
) -> dict:
    """
    用于审核两张图片是否是在相同的背景下拍摄，综合了VLM和传统计算机视觉方法
    输出一个结果字典，包含"判断依据"和"审查结果"两个字段
    """
    image_url_1 = uploader.upload_and_get_presigned_url(Path(image_path_1),"image")
    image_url_2 = uploader.upload_and_get_presigned_url(Path(image_path_2),"image")
    print(f'图片上传云端结束')
    cv_report = cv_method(image_path_1, image_path_2)
    print(f'图片传统计算机视觉对比结束：\n{cv_report}')
    system_prompt = """
<role>
你是一位注册资料审查员，专门负责证件审查。
</role>

<task>
你将会收到两张身份证图片和一份对这两张图片的审查报告
请参考审查报告并结合你看到的图片，判断这两张图片是否在同一个背景下拍摄，只有拍摄于同一个背景才能通过
背景可以有旋转，位移，透视缩放，但是必须是同一个物体，比如同一本书，同一张桌子等
完成任务后请按照<output_format>中的格式要求输出，所需字段描述如下：
"图片背景描述"：描述你看到了什么背景，大概是什么东西，有什么特征
"判断依据"：基于"图片背景描述"和审查报告的内容，综合判断两张图片是否在同一个背景下拍摄
"审查结果"：该字段只能从["通过","不通过"]选择一个输出，指明两张图片是否在同一个背景拍摄的审核结论

除了要求输出的字典外不要有任何多余的语言
</task>

<output_format>
{
"图片背景描述":"XXXXXXXXXX..."
"判断依据":"XXXXXXX...",
"审查结果":"通过"
}
</output_format>

"""

    message_list = [
        SystemMessage(system_prompt),
        HumanMessage(
                [
            {
                'type': 'image_url',
                'image_url': {'url': image_url_1, 'detail': "high"}
            },
            {
                'type': 'image_url',
                'image_url': {'url': image_url_2, 'detail': "high"}
            }
            ]
        ),
        HumanMessage(cv_report)
    ]
    res = visual_llm.invoke(message_list)
    # 这里没做输出格式核验
    print(f'llm整合结束')
    return json5.loads(res.content)
