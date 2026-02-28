import cv2
import numpy as np
import json
from langchain_core.tools import tool
from pathlib import Path

# TODO需要review

def _mask_id_card(image: np.ndarray) -> np.ndarray:
    """
    精细版辅助函数：利用几何特征和宽高比，动态识别并遮挡身份证区域。
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
    # 身份证反光可能导致边缘断裂，这一步能将其重新连成一个完整的框
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

        # 安全操作：稍微向外膨胀一点黑色区域（腐蚀白色掩膜），
        # 确保身份证的边缘（甚至是手指捏着的地方）也被完全遮挡，避免边缘干扰特征点提取
        kernel_erode = np.ones((20, 20), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=1)
    else:
        # 【降级方案 (Fallback)】：如果背景过于杂乱或对比度太低导致找不到轮廓
        # 退回到保守的中央区域遮挡，防止程序崩溃
        x1, y1 = int(w * 0.25), int(h * 0.25)
        x2, y2 = int(w * 0.75), int(h * 0.75)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

    return mask


@tool
def verify_background_consistency(workspace_dir: str) -> str:
    """
    验证工作区内的两张证件照片背景是否属于同一个真实物理位置。
    输入：包含待验证照片的文件夹绝对或相对路径 (workspace_dir)。
    输出：包含 is_same_place, inliers_count, homography_valid, confidence_score 的 JSON 字符串。
    """
    try:
        # ==========================================
        # 1. 自动接管工作区：彻底消除大模型的路径幻觉
        # ==========================================
        work_path = Path(workspace_dir)
        if not work_path.exists() or not work_path.is_dir():
            return json.dumps({"error": f"工作区路径无效或不存在: {workspace_dir}"})

        # 智能扫描常见图片格式，忽略大小写
        valid_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            p for p in work_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]

        if len(image_files) < 2:
            return json.dumps({
                "error": f"工作区内图片不足2张，当前找到 {len(image_files)} 张，无法进行比对。请检查输入源。"
            })

        # 稳定抓取前两张图片（系统通常会为单次任务准备干净的独立文件夹）
        image_path_1 = str(image_files[0])
        image_path_2 = str(image_files[1])

        print(f"  [视觉引擎] 成功从工作区抓取目标: {image_files[0].name} & {image_files[1].name}")

        # ==========================================
        # 2. 核心视觉算法执行 (完全复用你的硬核逻辑)
        # ==========================================
        # 以灰度模式读取图片
        img1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return json.dumps({"error": "图片 OpenCV 读取失败，可能是文件损坏。"})

        # 生成遮罩，过滤掉证件主体
        mask1 = _mask_id_card(img1)
        mask2 = _mask_id_card(img2)

        # 提取 SIFT 特征点和描述子
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, mask1)
        kp2, des2 = sift.detectAndCompute(img2, mask2)

        # 如果背景过于干净（如纯白纸），直接短路返回
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return json.dumps({
                "is_same_place": False,
                "inliers_count": 0,
                "homography_valid": False,
                "confidence_score": 0.0
            })

        # 使用 FLANN 进行特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 使用 Lowe's ratio test 筛选高质量的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        inliers_count = 0
        homography_valid = False
        confidence_score = 0.0

        # 空间透视一致性校验 (Homography + RANSAC)
        if len(good_matches) >= 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                homography_valid = True
                inliers_count = int(np.sum(mask))
                confidence_score = round(inliers_count / max(len(good_matches), 1), 2)
        else:
            inliers_count = len(good_matches)

        # 综合判定
        is_same_place = bool(homography_valid and inliers_count >= 15 and confidence_score >= 0.85)

        # 组装结果并返回
        result = {
            "is_same_place": is_same_place,
            "inliers_count": inliers_count,
            "homography_valid": homography_valid,
            "confidence_score": confidence_score
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"图像处理过程中发生异常: {str(e)}"})