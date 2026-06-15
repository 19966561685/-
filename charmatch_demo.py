#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
charmatch_demo.py — 字符级模板匹配算法验证 Demo  (v2)
==========================================================================
基于 PySide6 + OpenCV，实现传统字符级模板匹配算法。
算法核心参考自 C++ charmatch 插件的仿射变换、坐标映射反切、OK/NG 双列表判定。
字符切分采用基于轮廓的空间距离聚类，兼容中文字符偏旁部首合并及弧形排布。

【Phase A — 基准模板建立】
  1. 加载一张模板图片
  2. 在图上绘制【定位框(蓝色)】和【检测框(黄色)】
  3. 「设为基准模板」— 定位框纹理作为全局匹配基准，检测框位置存为相对于定位框的偏移
  4. 调整右侧「分割参数」— 检测框内实时显示绿色字符切分框（每个字符一个绿框）
  5. 「保存字符模板图片」— 将切好的单字 tpl_行_列.png 保存到自定义文件夹

【Phase B — 在线检测】
  1. 「切换下一张」— 加载一张新图（模拟相机抓拍）
  2. 模板匹配找到定位框在新图中的位置
  3. ROI联动：检测框 = 定位框新位置 + 相对偏移
  4. 在新的检测框内切分字符，逐一与保存的字符模板做 matchTemplate 比对
  5. 全匹配 → 绿色框覆盖每个字符，UI 显示绿色 OK
  6. 缺少字符/不匹配 → 该字符显示红色框，UI 显示红色 NG
  7. 调整「匹配阈值」控制检测精度

【关键算法 — 轮廓空间距离聚类】
  替代 C++ 的水平行列投影法，采用图连通分量(BFS)对邻近轮廓进行空间聚类：
  - 计算每个轮廓的 boundingRect
  - 两轮廓水平间距 < ε 且垂直间距 < ε×3 → 连通
  - BFS 找连通分量 → 合并为一个字符矩形
  - 中文字符偏旁部首(氵、扌、钅等)自动合并，弧形排布也兼容
==========================================================================
"""

import sys
import os
import json
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QCheckBox,
    QGroupBox, QGridLayout, QSplitter, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsItem,
    QScrollArea,
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, Signal, Slot,
)
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QBrush, QColor, QFont, QPainter,
    QCursor, QWheelEvent, QMouseEvent,
)


# ========================================================================
# 第一部分：核心算法模块（纯 OpenCV）
# ========================================================================

def mat_to_qimage(cv_mat):
    """将 OpenCV BGR/灰度图 转换为 QImage"""
    if cv_mat is None or cv_mat.size == 0:
        return QImage()
    if len(cv_mat.shape) == 2:
        h, w = cv_mat.shape
        return QImage(cv_mat.data, w, h, w, QImage.Format_Grayscale8).copy()
    elif cv_mat.shape[2] == 3:
        h, w, _ = cv_mat.shape
        rgb = cv2.cvtColor(cv_mat, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QImage()


def adaptive_threshold(gray, block_size, invert=False):
    """自适应二值化，对应 C++ adaptiveThreshold(…, ADAPTIVE_THRESH_MEAN_C, …)"""
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    c_param = block_size if invert else -block_size
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  thresh_type, block_size, c_param)


def morphological_close(bin_img, kx, ky):
    """形态学闭运算（连接断裂笔画）"""
    if kx <= 0 or ky <= 0:
        return bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)


def cluster_characters_by_distance(contours, eps, max_width):
    """
    ============================================================
    基于轮廓空间距离聚类的字符切分（核心算法）
    ============================================================
    替代 C++ 版本的水平行列投影法，采用图连通分量（graph-connected
    components）对邻近轮廓进行空间聚类。

    步骤：
      a) 对每个轮廓求 boundingRect
      b) 计算两两之间的水平和垂直间距
      c) 水平间距 < eps 且 垂直间距 < eps*3 → 连通
      d) BFS 找连通分量，每个分量 = 一个字符
      e) 合并分量内所有轮廓的外接矩形 → 字符切片矩形
      f) 按 (y, x) 排序输出
    ============================================================
    """
    if not contours:
        return []

    rects = [cv2.boundingRect(c) for c in contours]
    n = len(rects)
    if n == 0:
        return []

    adj = [[] for _ in range(n)]
    for i in range(n):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = rects[j]
            h_gap = max(0, max(xi - (xj + wj), xj - (xi + wi)))
            v_gap = max(0, max(yi - (yj + hj), yj - (yi + hi)))
            if h_gap < eps and v_gap < eps * 3:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        cluster_rects = []
        while stack:
            idx = stack.pop()
            cluster_rects.append(rects[idx])
            for nb in adj[idx]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(cluster_rects)

    merged = []
    for cr in clusters:
        x_min = min(r[0] for r in cr)
        y_min = min(r[1] for r in cr)
        x_max = max(r[0] + r[2] for r in cr)
        y_max = max(r[1] + r[3] for r in cr)
        cw = x_max - x_min
        ch = y_max - y_min
        if cw <= max_width and cw > 2 and ch > 2:
            merged.append((x_min, y_min, cw, ch))

    merged.sort(key=lambda r: (r[1], r[0]))
    return merged


def detect_characters(gray, params):
    """字符检测主流程，返回 (char_rects, bin_img)"""
    bin_img = adaptive_threshold(gray, params['block_size'], params['invert'])
    morph_img = morphological_close(bin_img, params['ksize_x'], params['ksize_y'])
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    min_area = params['min_area']
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    char_rects = cluster_characters_by_distance(
        filtered, params['eps'], params['max_width']
    )
    return char_rects, bin_img


def trim_to_foreground(bin_img, rect, invert=False):
    """
    在二值图上找到 blob 的实际前景边界，返回更紧致的矩形。
    """
    x, y, w, h = rect
    roi = bin_img[y:y + h, x:x + w]
    if roi.size == 0:
        return rect
    if invert:
        fg_mask = roi == 0
    else:
        fg_mask = roi == 255
    fg_coords = np.column_stack(np.where(fg_mask))
    if len(fg_coords) == 0:
        return rect
    ty_min = int(fg_coords[:, 0].min())
    ty_max = int(fg_coords[:, 0].max())
    tx_min = int(fg_coords[:, 1].min())
    tx_max = int(fg_coords[:, 1].max())
    tw = max(tx_max - tx_min + 1, 2)
    th = max(ty_max - ty_min + 1, 2)
    return (x + tx_min, y + ty_min, tw, th)


def cluster_into_rows(rects, row_height_ratio=0.6):
    """
    将字符矩形按行聚类，生成行列编号。
    返回 [(x, y, w, h, row, col), ...]
    """
    if not rects:
        return []
    centers_y = [(r[1] + r[3] // 2) for r in rects]
    sorted_indices = sorted(range(len(rects)), key=lambda i: centers_y[i])
    heights = [rects[i][3] for i in sorted_indices]
    median_h = np.median(heights) if heights else 20
    row_threshold = median_h * row_height_ratio
    rows = []
    current_row = [sorted_indices[0]]
    current_y = centers_y[sorted_indices[0]]
    for idx in sorted_indices[1:]:
        cy = centers_y[idx]
        if cy - current_y < row_threshold:
            current_row.append(idx)
        else:
            rows.append(current_row)
            current_row = [idx]
            current_y = cy
    if current_row:
        rows.append(current_row)
    result = []
    for row_idx, row_indices in enumerate(rows):
        row_sorted = sorted(row_indices, key=lambda i: rects[i][0] + rects[i][2] // 2)
        for col_idx, orig_idx in enumerate(row_sorted):
            x, y, w, h = rects[orig_idx]
            result.append((x, y, w, h, row_idx, col_idx))
    return result


def match_template_score(tpl, target):
    """
    TM_CCOEFF_NORMED 模板匹配，返回 [0, 1] 得分。
    增强: 当模板略大于目标时自动缩放适应，避免返回 0。
    """
    if tpl is None or target is None or tpl.size == 0 or target.size == 0:
        return 0.0
    th, tw = tpl.shape[:2]
    dh, dw = target.shape[:2]
    # 模板比目标大时缩放到目标尺寸
    if th > dh or tw > dw:
        scale = min(dh / max(th, 1), dw / max(tw, 1))
        if scale < 0.3:
            return 0.0
        new_w = max(int(tw * scale), 2)
        new_h = max(int(th * scale), 2)
        try:
            tpl = cv2.resize(tpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error:
            return 0.0
    th2, tw2 = tpl.shape[:2]
    if th2 > dh or tw2 > dw:
        return 0.0
    try:
        result = cv2.matchTemplate(target, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        # TM_CCOEFF_NORMED 范围 [-1, 1]，负值表示反相关，clamp 到 [0, 1]
        return max(0.0, float(max_val))
    except cv2.error:
        return 0.0


def align_by_template_match(src_gray, templ_gray, template_rect):
    """全局模板匹配定位，返回 (best_x, best_y, score)"""
    x, y, w, h = template_rect
    th, tw = templ_gray.shape[:2]
    x = max(0, min(x, tw - 1))
    y = max(0, min(y, th - 1))
    w = min(w, tw - x)
    h = min(h, th - y)
    if w <= 0 or h <= 0:
        return None, None, 0.0
    templ_roi = templ_gray[y:y + h, x:x + w]
    sh, sw = src_gray.shape[:2]
    if sh < h or sw < w:
        return None, None, 0.0
    result = cv2.matchTemplate(src_gray, templ_roi, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return int(max_loc[0]), int(max_loc[1]), float(max_val)


def align_by_feature_matching(src_gray, templ_gray, template_rect,
                             search_rect=None, nfeatures=800):
    """
    基于 ORB 特征匹配的旋转不变定位。

    当模板匹配无法应对图像旋转时，用 ORB 特征点找到旋转角度和变换矩阵。

    Args:
        src_gray:     检测图（灰度）
        templ_gray:    模板图（灰度）
        template_rect:  模板图上定位框 (x, y, w, h)
        search_rect:    可选，在检测图上限定搜索区域 (x, y, w, h)，加速匹配
        nfeatures:      ORB 特征点数量

    Returns:
        (cx, cy, angle_deg, score, M_affine)  或  (None, None, 0.0, 0.0, None)
        cx, cy:    定位框中心在检测图中的坐标
        angle_deg:  定位框在检测图中的旋转角度（弧度，正值=逆时针）
        M_affine:   2×3 仿射矩阵（模板全图坐标 → 检测图坐标）
    """
    x, y, w, h = template_rect
    th, tw = templ_gray.shape[:2]
    x = max(0, min(x, tw - 1))
    y = max(0, min(y, th - 1))
    w = min(w, tw - x)
    h = min(h, th - y)
    if w <= 0 or h <= 0:
        return None, None, 0.0, 0.0, None

    tpl_patch = templ_gray[y:y + h, x:x + w]
    if tpl_patch.size == 0 or tpl_patch.shape[0] < 10 or tpl_patch.shape[1] < 10:
        return None, None, 0.0, 0.0, None

    # ---- 限定搜索区域 ----
    if search_rect is not None:
        sx, sy, sw, sh = search_rect
        sh_img, sw_img = src_gray.shape[:2]
        sx = max(0, min(sx, sw_img - 1))
        sy = max(0, min(sy, sh_img - 1))
        sw = min(sw, sw_img - sx)
        sh = min(sh, sh_img - sy)
        if sw >= 10 and sh >= 10:
            src_roi = src_gray[sy:sy + sh, sx:sx + sw]
        else:
            src_roi = src_gray
            sx, sy = 0, 0
    else:
        src_roi = src_gray
        sx, sy = 0, 0

    # ---- ORB 特征检测 ----
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(tpl_patch, None)

    if search_rect is not None:
        kp2, des2 = orb.detectAndCompute(src_roi, None)
        # 将 ROI 内的关键点坐标偏移回全图坐标
        for kp in kp2:
            kp.pt = (kp.pt[0] + sx, kp.pt[1] + sy)
    else:
        kp2, des2 = orb.detectAndCompute(src_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None, 0.0, 0.0, None

    # ---- BFMatcher 匹配 ----
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw_matches = bf.match(des1, des2)
    if len(raw_matches) < 6:
        return None, None, 0.0, 0.0, None
    raw_matches = sorted(raw_matches, key=lambda m: m.distance)

    # 筛选较优匹配（距离 < 70 或前30个）
    good_matches = [m for m in raw_matches if m.distance < 70]
    if len(good_matches) < 6:
        good_matches = raw_matches[:min(30, len(raw_matches))]
    if len(good_matches) < 6:
        return None, None, 0.0, 0.0, None

    # src_pts 需要从 patch 局部坐标偏移到模板全图坐标
    src_pts_list = []
    for m in good_matches:
        px, py = kp1[m.queryIdx].pt
        src_pts_list.append([px + x, py + y])
    src_pts = np.float32(src_pts_list).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ---- RANSAC 求仿射变换矩阵（允许旋转+平移，不允许透视变形）----
    # 用 estimateAffinePartial2D 代替 findHomography，更直接地得到旋转角度
    M_affine, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M_affine is None:
        return None, None, 0.0, 0.0, None

    n_inliers = int(np.sum(mask)) if mask is not None else len(good_matches)
    score = min(n_inliers / max(len(good_matches), 1), 1.0)
    if n_inliers < 4 or score < 0.15:
        return None, None, 0.0, 0.0, None

    # ---- 从仿射矩阵提取旋转角度和中心点 ----
    # M_affine: [[cosθ, -sinθ, tx],
    #             [sinθ,  cosθ, ty]]
    cos_a = M_affine[0, 0]
    sin_a = M_affine[1, 0]
    angle_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))  # 归一化到 [-180, 180]

    # 模板定位框中心
    tpl_cx = x + w / 2.0
    tpl_cy = y + h / 2.0

    # 变换后中心 = M * [tpl_cx, tpl_cy, 1]^T
    det_cx = M_affine[0, 0] * tpl_cx + M_affine[0, 1] * tpl_cy + M_affine[0, 2]
    det_cy = M_affine[1, 0] * tpl_cx + M_affine[1, 1] * tpl_cy + M_affine[1, 2]

    return float(det_cx), float(det_cy), angle_deg, min(score, 1.0), M_affine


def _transform_rect(M, x, y, w, h):
    """用仿射变换矩阵 M 变换矩形的四个角点，返回变换后的角点列表。"""
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    ones = np.ones((4, 1), dtype=np.float64)
    corners_h = np.hstack([corners, ones])  # 4×3
    transformed = (M @ corners_h.T).T  # 4×2
    return transformed


def extract_rotated_rect(img_gray, rotated_rect):
    """
    从原图中提取旋转矩形的摆正后图像（rectified）。

    rotated_rect: ((cx, cy), (w, h), angle) — OpenCV RotatedRect 格式

    Returns:
        rectified: 摆正后的图像 (h×w)，其中矩形区域变为无旋转的轴对齐图像
        M_d2s:     2×3 仿射矩阵，将摆正图坐标映射回原图坐标
                   src_pt = M_d2s @ [rect_x, rect_y, 1]^T
    """
    center, size, angle = rotated_rect
    w, h = int(size[0]), int(size[1])
    cx, cy = center
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    # M 将 rectified 坐标映射回原图坐标（也即 warpAffine 所需的 M）
    M = np.array([
        [cos_a, -sin_a, cx - cos_a * w / 2 + sin_a * h / 2],
        [sin_a,  cos_a, cy - sin_a * w / 2 - cos_a * h / 2]
    ], dtype=np.float64)

    rectified = cv2.warpAffine(img_gray, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return rectified, M


def _draw_rotated_rect(img, center, size, angle_deg, color, thickness=2):
    """在 OpenCV 图像上绘制旋转矩形。"""
    box = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), angle_deg))
    cv2.polylines(img, [np.int32(box)], True, color, thickness, lineType=cv2.LINE_AA)


def _draw_rotated_box(img, center, size, angle_deg, color, is_ok, label="", thickness=2):
    """绘制一个旋转的检测框（OK绿/NG红），可选带标签。"""
    fill_color = tuple(int(c * 0.3) for c in color)  # 半透明填充
    box = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), angle_deg))
    box = np.int32(box)
    # 填充
    overlay = img.copy()
    cv2.fillPoly(overlay, [box], fill_color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    # 边框
    cv2.polylines(img, [box], True, color, thickness, lineType=cv2.LINE_AA)
    # 标签
    if label:
        tx, ty = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2 - 4)
        if ty < 5:
            ty = int(center[1] + size[1] / 2 + 14)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ========================================================================
# 第二部分：可拖拽/可调整大小的 ROI 矩形项
# ========================================================================

class RoiRectItem(QGraphicsRectItem):
    """
    可拖拽、可调整大小的 ROI 矩形。
    - 内部拖拽 → 整体移动
    - 四角拖拽 → 同时调整宽高
    - 四边拖拽 → 独立调整宽度或高度（新增）
    - 光标在角/边自动变为对应缩放箭头
    """

    CORNER_MARGIN = 15
    EDGE_MARGIN = 10
    MIN_SIZE = 30

    def __init__(self, rect: QRectF, color: QColor, label: str = ""):
        super().__init__(rect)
        self._color = color
        self._label = label
        fill = QColor(color)
        fill.setAlpha(35)
        self.setBrush(QBrush(fill))
        self.setPen(QPen(color, 2))
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self._interacting = False
        self._interact_mode = None
        self._interact_start_pos = None   # 鼠标按下时的场景坐标
        self._interact_start_rect = None  # 鼠标按下时的矩形（本地坐标）
        self._drag_start_pos = None       # 鼠标按下时的 item 场景位置
        self._bounds_rect = None          # 可选：场景坐标系中的限制矩形（x,y,w,h）
        self._on_resized = None           # 可选：拖拽/缩放结束后的回调函数

    def _hit_test(self, pos):
        """检测鼠标位置: 'move'|'tl'|'tr'|'br'|'bl'|'left'|'right'|'top'|'bottom'|None"""
        r = self.rect()
        x, y, w2, h2 = r.x(), r.y(), r.width(), r.height()
        rx, ry = x + w2, y + h2
        cm, em = self.CORNER_MARGIN, self.EDGE_MARGIN
        if abs(pos.x() - x) < cm and abs(pos.y() - y) < cm: return 'tl'
        if abs(pos.x() - rx) < cm and abs(pos.y() - y) < cm: return 'tr'
        if abs(pos.x() - rx) < cm and abs(pos.y() - ry) < cm: return 'br'
        if abs(pos.x() - x) < cm and abs(pos.y() - ry) < cm: return 'bl'
        if abs(pos.x() - x) < em and y < pos.y() < ry: return 'left'
        if abs(pos.x() - rx) < em and y < pos.y() < ry: return 'right'
        if abs(pos.y() - y) < em and x < pos.x() < rx: return 'top'
        if abs(pos.y() - ry) < em and x < pos.x() < rx: return 'bottom'
        if r.contains(pos): return 'move'
        return None

    def hoverMoveEvent(self, event):
        hit = self._hit_test(event.pos())
        cmap = {'move': Qt.SizeAllCursor,
                'tl': Qt.SizeFDiagCursor, 'br': Qt.SizeFDiagCursor,
                'tr': Qt.SizeBDiagCursor, 'bl': Qt.SizeBDiagCursor,
                'left': Qt.SizeHorCursor, 'right': Qt.SizeHorCursor,
                'top': Qt.SizeVerCursor, 'bottom': Qt.SizeVerCursor}
        self.setCursor(cmap.get(hit, Qt.ArrowCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            hit = self._hit_test(event.pos())
            if hit is not None:
                self._interacting = True
                self._interact_mode = hit
                self._interact_start_pos = event.scenePos()
                self._interact_start_rect = QRectF(self.rect())
                self._drag_start_pos = QPointF(self.pos())  # 记录 item 当前位置
                return
        super().mousePressEvent(event)

    def set_bounds(self, bx, by, bw, bh):
        """设置场景坐标系中的限制矩形，拖拽/缩放不能超出此范围。"""
        self._bounds_rect = (bx, by, bw, bh)

    def _clamp_to_bounds(self):
        """将 item 在当前 pos+rect 下 clamp 到 _bounds_rect 内。"""
        if self._bounds_rect is None:
            return
        bx, by, bw, bh = self._bounds_rect
        r = self.rect()
        # 场景坐标系中的当前矩形
        left = self.pos().x() + r.x()
        top = self.pos().y() + r.y()
        right = left + r.width()
        bottom = top + r.height()
        # 分别 clamp
        if left < bx:
            dx = bx - left
            self.setPos(self.pos().x() + dx, self.pos().y())
        if top < by:
            dy = by - top
            self.setPos(self.pos().x(), self.pos().y() + dy)
        if right > bx + bw:
            dx = (bx + bw) - right
            self.setPos(self.pos().x() + dx, self.pos().y())
        if bottom > by + bh:
            dy = (by + bh) - bottom
            self.setPos(self.pos().x(), self.pos().y() + dy)
        # 重新读取可能被 pos 调整影响后的值
        left = self.pos().x() + r.x()
        top = self.pos().y() + r.y()
        right = left + r.width()
        bottom = top + r.height()
        # 如果尺寸超出边界，缩小尺寸
        if right > bx + bw:
            new_w = (bx + bw) - left
            r.setWidth(max(self.MIN_SIZE, new_w))
        if bottom > by + bh:
            new_h = (by + bh) - top
            r.setHeight(max(self.MIN_SIZE, new_h))
        if left < bx:
            r.setX(r.x() + (bx - left))
        if top < by:
            r.setY(r.y() + (by - top))
        self.setRect(r)

    def mouseMoveEvent(self, event):
        if not self._interacting or self._interact_start_pos is None:
            super().mouseMoveEvent(event)
            return

        # 计算相对于按下位置的偏移量（场景坐标系）
        sp = self._interact_start_pos
        delta = event.scenePos() - sp
        dx, dy = delta.x(), delta.y()

        mode = self._interact_mode

        # ---------- 移动模式 ----------
        if mode == 'move':
            if self._drag_start_pos is None:
                return
            new_pos = self._drag_start_pos + delta
            self.setPos(new_pos)
            self._clamp_to_bounds()
            return

        # ---------- 缩放模式 ----------
        r = self._interact_start_rect
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()

        if mode == 'tl':     x1 += dx; y1 += dy
        elif mode == 'tr':   x2 += dx; y1 += dy
        elif mode == 'br':   x2 += dx; y2 += dy
        elif mode == 'bl':   x1 += dx; y2 += dy
        elif mode == 'left':   x1 += dx
        elif mode == 'right':  x2 += dx
        elif mode == 'top':    y1 += dy
        elif mode == 'bottom': y2 += dy

        nx, ny = min(x1, x2), min(y1, y2)
        nw, nh = abs(x2 - x1), abs(y2 - y1)
        if nw >= self.MIN_SIZE and nh >= self.MIN_SIZE:
            self.setRect(QRectF(nx, ny, nw, nh))
            self._clamp_to_bounds()

    def set_on_resized(self, callback):
        """设置回调，在拖拽/缩放结束后调用。"""
        self._on_resized = callback

    def mouseReleaseEvent(self, event):
        self._interacting = False
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)
        if self._on_resized:
            self._on_resized()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self._label:
            painter.setPen(QPen(self._color, 1))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(self.rect().adjusted(5, 3, -5, -3),
                             Qt.AlignLeft | Qt.AlignTop, self._label)

    def get_state(self):
        return {'rect': QRectF(self.rect()), 'pos': QPointF(self.pos())}

    @staticmethod
    def from_state(state, color, label):
        item = RoiRectItem(state['rect'], color, label)
        item.setPos(state['pos'])
        return item

    def get_roi(self):
        r = self.rect()
        p = self.pos()
        return (int(p.x() + r.x()), int(p.y() + r.y()),
                int(r.width()), int(r.height()))


# ========================================================================
# 第三部分：主窗口
# ========================================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CharMatch Demo")
        self.resize(1450, 920)
        self._img_bgr = None
        self._img_gray = None
        self._tpl_bgr = None
        self._tpl_gray = None
        self._template_set = False
        self._chars_detected = False
        self._char_templates = []
        self._save_dir = ""
        self._align_ref_rect = None
        self._detect_ref_rect = None
        self._region_ref_rect = None
        self._detect_offset = (0, 0)
        self._current_char_overlays = []
        self._params = {
            'block_size': 13, 'invert': False,
            'ksize_x': 5, 'ksize_y': 3,
            'max_width': 90, 'eps': 5, 'min_area': 20,
            'margin': 0, 'match_thresh': 80,
        }
        self._build_ui()
        self._connect_signals()
        self._auto_load_params()
        self._update_ui_state()

    def _auto_load_params(self):
        fpath = self._param_file_path()
        if os.path.isfile(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                self._params.update(loaded)
                self._apply_params_to_ui()
            except Exception:
                pass

    # ==================== UI ====================

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QHBoxLayout(central)
        ml.setContentsMargins(6, 6, 6, 6)
        ml.setSpacing(6)

        # View
        self._scene = QGraphicsScene()
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setStyleSheet("background:#2a2a2a; border:1px solid #555;")
        self._placeholder = self._scene.addText(
            "请先加载一张模板图片\n\n鼠标滚轮缩放 | 拖拽空白区平移\n拖拽 ROI 矩形移动 | 拖拽角/边缩放",
            QFont("Arial", 13))
        self._placeholder.setDefaultTextColor(QColor(160, 160, 160))
        self._placeholder.setPos(60, 100)
        self._pixmap_item = None
        self._align_roi = None
        self._region_roi = None
        self._detect_roi = None
        self._overlay_items = []

        self._status_label = QLabel(self._view)
        self._status_label.setStyleSheet(
            "background:rgba(0,0,0,160); color:#00ff88; "
            "font-size:28px; font-weight:bold; padding:8px 20px; border-radius:6px;")
        self._status_label.setText("就绪")
        self._status_label.adjustSize()
        self._status_label.move(20, 20)

        # Right panel
        rp = QWidget()
        rl = QVBoxLayout(rp)
        rl.setSpacing(8)
        rl.setContentsMargins(4, 4, 4, 4)

        def mk_spin(lo, hi, default, step=1):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(default); s.setSingleStep(step); return s

        # Buttons
        bg = QGroupBox("操作流程")
        bl = QVBoxLayout(bg)
        self._btn_load_tpl = QPushButton("① 加载模板图片")
        self._btn_set_tpl = QPushButton("② 设为基准模板")
        self._btn_save_chars = QPushButton("③ 保存字符模板图片")
        self._btn_next = QPushButton("④ 切换下一张 (检测)")
        self._cb_region_roi = QCheckBox("启用区域ROI框")
        for b in [self._btn_load_tpl, self._btn_set_tpl, self._btn_save_chars, self._btn_next]:
            b.setMinimumHeight(34); b.setFont(QFont("Arial", 10))
        for b in [self._btn_load_tpl, self._btn_set_tpl, self._btn_save_chars, self._btn_next]:
            bl.addWidget(b)
        bl.addWidget(self._cb_region_roi)
        rl.addWidget(bg)

        # Params
        pg = QGroupBox("分割参数（调整后实时预览字符切分）")
        pl = QGridLayout(pg)
        pl.setVerticalSpacing(5); pl.setHorizontalSpacing(6)
        r = 0
        pl.addWidget(QLabel("二值化强度:"), r, 0)
        self._sp_bs = mk_spin(3, 99, 13, 2); pl.addWidget(self._sp_bs, r, 1); r += 1
        pl.addWidget(QLabel("反转二值化:"), r, 0)
        self._cb_inv = QCheckBox(); pl.addWidget(self._cb_inv, r, 1); r += 1
        pl.addWidget(QLabel("形态学X (宽度):"), r, 0)
        self._sp_kx = mk_spin(0, 300, 5); pl.addWidget(self._sp_kx, r, 1); r += 1
        pl.addWidget(QLabel("形态学Y (高度):"), r, 0)
        self._sp_ky = mk_spin(0, 200, 3); pl.addWidget(self._sp_ky, r, 1); r += 1
        pl.addWidget(QLabel("字符最大宽:"), r, 0)
        self._sp_mw = mk_spin(10, 500, 90); pl.addWidget(self._sp_mw, r, 1); r += 1
        pl.addWidget(QLabel("字符间距 (合并):"), r, 0)
        self._sp_eps = mk_spin(0, 60, 5); pl.addWidget(self._sp_eps, r, 1); r += 1
        pl.addWidget(QLabel("最小面积:"), r, 0)
        self._sp_ma = mk_spin(1, 999, 20); pl.addWidget(self._sp_ma, r, 1); r += 1
        pl.addWidget(QLabel("匹配边距:"), r, 0)
        self._sp_margin = mk_spin(0, 60, 0); pl.addWidget(self._sp_margin, r, 1); r += 1
        rl.addWidget(pg)

        tg = QGroupBox("检测参数")
        tl = QGridLayout(tg)
        tl.addWidget(QLabel("匹配阈值 (0~100):"), 0, 0)
        self._sp_th = mk_spin(0, 100, 80); tl.addWidget(self._sp_th, 0, 1)
        rl.addWidget(tg)

        psg = QGroupBox("参数持久化")
        psl = QHBoxLayout(psg)
        self._btn_save_params = QPushButton("保存参数")
        self._btn_load_params = QPushButton("加载参数")
        for b in [self._btn_save_params, self._btn_load_params]:
            b.setMinimumHeight(28); psl.addWidget(b)
        rl.addWidget(psg)

        pv = QGroupBox("二值化预览 (检测框ROI)")
        pvl = QVBoxLayout(pv)
        self._bin_preview_label = QLabel("等待检测框...")
        self._bin_preview_label.setAlignment(Qt.AlignCenter)
        self._bin_preview_label.setMinimumHeight(140)
        self._bin_preview_label.setStyleSheet(
            "background:#1a1a1a; color:#888; border:1px solid #555; font-size:11px;")
        self._bin_preview_label.setScaledContents(True)
        pvl.addWidget(self._bin_preview_label)
        rl.addWidget(pv)

        self._info = QLabel("就绪，请加载模板图片")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("background:#f0f0f0; padding:8px; border-radius:4px;")
        self._info.setMinimumHeight(80)
        self._info.setAlignment(Qt.AlignTop)
        rl.addWidget(self._info)
        rl.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(rp)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea{border:none;}")

        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(self._view)
        sp.addWidget(scroll)
        sp.setStretchFactor(0, 3)
        sp.setStretchFactor(1, 1)
        ml.addWidget(sp)

    def _connect_signals(self):
        self._btn_load_tpl.clicked.connect(self._on_load_template)
        self._btn_set_tpl.clicked.connect(self._on_set_template)
        self._btn_save_chars.clicked.connect(self._on_save_chars)
        self._btn_next.clicked.connect(self._on_next_image)
        for sp in [self._sp_bs, self._sp_kx, self._sp_ky,
                    self._sp_mw, self._sp_eps, self._sp_ma, self._sp_margin, self._sp_th]:
            sp.valueChanged.connect(self._on_param_changed)
        self._cb_inv.stateChanged.connect(self._on_param_changed)
        self._btn_save_params.clicked.connect(self._on_save_params)
        self._btn_load_params.clicked.connect(self._on_load_params)
        self._cb_region_roi.stateChanged.connect(self._on_region_roi_toggled)

    def _on_region_roi_toggled(self):
        """区域ROI框勾选状态改变时：如果模板已设置则提示重置，否则刷新ROI显示"""
        if self._img_bgr is None:
            return
        if self._template_set:
            self._template_set = False
            self._align_ref_rect = None
            self._detect_ref_rect = None
            self._region_ref_rect = None
            self._char_templates = []
            self._chars_detected = False
            self._status_label.setText("区域ROI变更，请重新设置基准模板")
            self._status_label.setStyleSheet(
                "background:rgba(0,0,0,160); color:#ffcc00; font-size:18px; font-weight:bold; padding:6px 16px; border-radius:6px;")
        self._create_default_rois()

    def _update_align_bounds(self):
        """当区域ROI被拖拽/缩放后，更新定位框的边界限制。"""
        if self._region_roi and self._align_roi:
            bx, by, bw, bh = self._region_roi.get_roi()
            self._align_roi.set_bounds(bx, by, bw, bh)
            # 将定位框 clamp 到新边界内
            self._align_roi._clamp_to_bounds()

    def _update_ui_state(self):
        has_img = self._img_bgr is not None
        has_tpl = self._template_set
        has_chars = len(self._char_templates) > 0
        self._btn_load_tpl.setEnabled(True)
        self._btn_set_tpl.setEnabled(has_img)
        self._btn_save_chars.setEnabled(has_tpl and self._chars_detected)
        self._btn_next.setEnabled(has_tpl and has_chars)
        self._cb_region_roi.setEnabled(has_img)

    def _save_roi_state(self):
        a = self._align_roi.get_state() if self._align_roi else None
        d = self._detect_roi.get_state() if self._detect_roi else None
        r = self._region_roi.get_state() if self._region_roi else None
        return a, d, r

    def _restore_roi(self, state, color, label):
        if state is None: return None
        item = RoiRectItem.from_state(state, color, label)
        self._scene.addItem(item)
        return item

    def _show_image(self, bgr_img, overlays=None, show_rois=True):
        if bgr_img is None: return
        a_state, d_state, r_state = self._save_roi_state()
        h, w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) if bgr_img.ndim == 3 else bgr_img
        qimg = QImage(rgb.data, w, h, w * (3 if rgb.ndim == 3 else 1),
                      QImage.Format_RGB888 if rgb.ndim == 3 else QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self._scene.clear()
        self._placeholder = None
        self._overlay_items = []
        self._align_roi = None
        self._region_roi = None
        self._detect_roi = None
        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(0, 0, w, h)
        if show_rois:
            if r_state:
                self._region_roi = self._restore_roi(r_state, QColor(160, 80, 220), "区域ROI(紫)")
            self._align_roi = self._restore_roi(a_state, QColor(60, 120, 255), "定位框(蓝)")
            self._detect_roi = self._restore_roi(d_state, QColor(255, 200, 40), "检测框(黄)")
            if self._region_roi:
                bx, by, bw, bh = self._region_roi.get_roi()
                self._align_roi.set_bounds(bx, by, bw, bh)
                self._region_roi.set_on_resized(self._update_align_bounds)
        if overlays:
            for item in overlays:
                x, y, ow, oh, is_ok = item[:5]
                label = item[5] if len(item) > 5 else ""
                color = QColor(0, 200, 0) if is_ok else QColor(200, 0, 0)
                fill = QColor(color); fill.setAlpha(80 if is_ok else 100)
                ri = QGraphicsRectItem(x, y, ow, oh)
                ri.setBrush(QBrush(fill))
                pen = QPen(color, 2); pen.setStyle(Qt.SolidLine if is_ok else Qt.DashLine)
                ri.setPen(pen); ri.setZValue(20)
                self._scene.addItem(ri); self._overlay_items.append(ri)
                if label:
                    txt = self._scene.addText(label, QFont("Arial", 8))
                    txt.setDefaultTextColor(Qt.white if is_ok else Qt.yellow)
                    txt.setPos(x + 2, y); txt.setZValue(21)

    def _create_default_rois(self):
        if self._img_bgr is None: return
        h, w = self._img_bgr.shape[:2]
        if self._align_roi: self._scene.removeItem(self._align_roi)
        if self._detect_roi: self._scene.removeItem(self._detect_roi)
        if self._region_roi:
            self._scene.removeItem(self._region_roi)
            self._region_roi = None
        if self._cb_region_roi.isChecked():
            self._region_roi = RoiRectItem(QRectF(10, 10, int(w * 0.50), int(h * 0.70)),
                                            QColor(160, 80, 220), "区域ROI(紫)")
            self._scene.addItem(self._region_roi)
        self._align_roi = RoiRectItem(QRectF(20, int(h*0.18), int(w*0.28), int(h*0.22)),
                                       QColor(60, 120, 255), "定位框(蓝)")
        self._detect_roi = RoiRectItem(QRectF(int(w*0.44), int(h*0.36), int(w*0.48), int(h*0.28)),
                                        QColor(255, 200, 40), "检测框(黄)")
        self._scene.addItem(self._align_roi); self._scene.addItem(self._detect_roi)
        # 如果区域ROI存在，将定位框限制在区域框内
        if self._region_roi:
            bx, by, bw, bh = self._region_roi.get_roi()
            self._align_roi.set_bounds(bx, by, bw, bh)
            self._region_roi.set_on_resized(self._update_align_bounds)

    def _read_params(self):
        self._params['block_size'] = self._sp_bs.value()
        self._params['invert'] = self._cb_inv.isChecked()
        self._params['ksize_x'] = self._sp_kx.value()
        self._params['ksize_y'] = self._sp_ky.value()
        self._params['max_width'] = self._sp_mw.value()
        self._params['eps'] = self._sp_eps.value()
        self._params['min_area'] = self._sp_ma.value()
        self._params['margin'] = self._sp_margin.value()
        self._params['match_thresh'] = self._sp_th.value()

    def _apply_params_to_ui(self):
        for w in [self._sp_bs, self._sp_kx, self._sp_ky, self._sp_mw,
                   self._sp_eps, self._sp_ma, self._sp_margin, self._sp_th]:
            w.blockSignals(True)
        self._cb_inv.blockSignals(True)
        self._sp_bs.setValue(self._params['block_size'])
        self._sp_kx.setValue(self._params['ksize_x'])
        self._sp_ky.setValue(self._params['ksize_y'])
        self._sp_mw.setValue(self._params['max_width'])
        self._sp_eps.setValue(self._params['eps'])
        self._sp_ma.setValue(self._params['min_area'])
        self._sp_margin.setValue(self._params['margin'])
        self._sp_th.setValue(self._params['match_thresh'])
        self._cb_inv.setChecked(self._params['invert'])
        for w in [self._sp_bs, self._sp_kx, self._sp_ky, self._sp_mw,
                   self._sp_eps, self._sp_ma, self._sp_margin, self._sp_th]:
            w.blockSignals(False)
        self._cb_inv.blockSignals(False)

    def _param_file_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "charmatch_params.json")

    def _on_save_params(self):
        self._read_params()
        try:
            with open(self._param_file_path(), 'w', encoding='utf-8') as f:
                json.dump(self._params, f, ensure_ascii=False, indent=2)
            self._info.setText(f"参数已保存")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _on_load_params(self):
        fpath = self._param_file_path()
        if not os.path.isfile(fpath):
            QMessageBox.information(self, "提示", "未找到参数配置文件")
            return
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            self._params.update(loaded)
            self._apply_params_to_ui()
            if self._template_set and self._img_bgr is not None:
                self._run_char_preview()
        except Exception as e:
            QMessageBox.warning(self, "加载失败", str(e))

    # ==================== Phase A ====================

    @staticmethod
    def _imread_unicode(path):
        if not os.path.isfile(path): return None
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    def _on_load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模板图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*.*)")
        if not path: return
        img = self._imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取：{path}")
            return
        self._template_set = False
        self._chars_detected = False
        self._char_templates = []
        self._align_ref_rect = None
        self._detect_ref_rect = None
        self._region_ref_rect = None
        self._img_bgr = img.copy()
        self._img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._tpl_bgr = img.copy()
        self._tpl_gray = self._img_gray.copy()
        self._show_image(img)
        self._create_default_rois()
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._bin_preview_label.setText("等待检测框...")
        self._bin_preview_label.setPixmap(QPixmap())
        self._status_label.setText("模板图已加载")
        self._status_label.setStyleSheet(
            "background:rgba(0,0,0,160); color:#88ccff; font-size:20px; font-weight:bold; padding:6px 16px; border-radius:6px;")
        if self._region_roi and self._cb_region_roi.isChecked():
            hint = "区域ROI(紫)已启用，定位只在框内搜索"
        else:
            hint = "区域ROI未启用，定位在全图搜索"
        self._info.setText(f"模板图：{os.path.basename(path)}\n尺寸：{img.shape[1]}×{img.shape[0]}\n{hint}\n请调整蓝色定位框和黄色检测框位置，然后点击「设为基准模板」")
        self._update_ui_state()

    def _on_set_template(self):
        if self._img_bgr is None:
            QMessageBox.warning(self, "提示", "请先加载模板图片"); return
        if self._align_roi is None or self._detect_roi is None:
            QMessageBox.warning(self, "提示", "请先绘制 ROI 框"); return
        self._read_params()
        self._align_ref_rect = self._align_roi.get_roi()
        ar = self._align_roi.get_roi()
        dr = self._detect_roi.get_roi()
        self._detect_offset = (dr[0] - ar[0], dr[1] - ar[1])
        self._detect_ref_rect = dr
        if self._region_roi and self._cb_region_roi.isChecked():
            self._region_ref_rect = self._region_roi.get_roi()
        else:
            self._region_ref_rect = None
        self._template_set = True
        self._run_char_preview()
        self._status_label.setText("基准模板已建立")
        self._status_label.setStyleSheet(
            "background:rgba(0,0,0,160); color:#00ff88; font-size:20px; font-weight:bold; padding:6px 16px; border-radius:6px;")
        if self._region_ref_rect:
            extra = f"\n区域ROI(紫)：{self._region_ref_rect}（定位限制在框内）"
        else:
            extra = "\n区域ROI：未启用（全图定位）"
        self._info.setText(f"基准模板已建立\n定位框位置：{self._align_ref_rect}\n检测框偏移：dx={self._detect_offset[0]}, dy={self._detect_offset[1]}{extra}\n调整分割参数，实时观察字符切分效果\n满意后点击「保存字符模板图片」")
        self._update_ui_state()

    # ==================== Preview ====================

    def _on_param_changed(self):
        self._read_params()
        try:
            with open(self._param_file_path(), 'w', encoding='utf-8') as f:
                json.dump(self._params, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if self._template_set and self._img_bgr is not None:
            self._run_char_preview()

    def _run_char_preview(self):
        if self._detect_roi is None or self._img_gray is None: return
        rx, ry, rw, rh = self._detect_roi.get_roi()
        h_img, w_img = self._img_gray.shape[:2]
        rx = max(0, min(rx, w_img - 5)); ry = max(0, min(ry, h_img - 5))
        rw = min(rw, w_img - rx); rh = min(rh, h_img - ry)
        if rw < 10 or rh < 10: return
        roi_gray = self._img_gray[ry:ry + rh, rx:rx + rw].copy()
        char_rects, bin_img = detect_characters(roi_gray, self._params)
        trimmed = [trim_to_foreground(bin_img, cr, self._params['invert']) for cr in char_rects]
        rc = cluster_into_rows(trimmed)
        self._update_bin_preview(bin_img)
        overlays = []
        for item in rc:
            cx, cy, cw, ch, row, col = item
            overlays.append((rx + cx, ry + cy, cw, ch, True, f"{row}_{col}"))
        self._chars_detected = len(rc) > 0
        display = self._img_bgr.copy()
        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (40, 200, 255), 2)
        self._show_image(display, overlays, show_rois=True)
        nrows = len(set(item[4] for item in rc)) if rc else 0
        labels = " ".join([f"{r[4]}_{r[5]}" for r in rc[:10]])
        if len(rc) > 10: labels += " ..."
        self._info.setText(f"字符切分预览 — {len(rc)} 个字符 ({nrows}行)\n编号: {labels}")
        self._update_ui_state()

    def _update_bin_preview(self, bin_img):
        if bin_img is None or bin_img.size == 0:
            self._bin_preview_label.setText("无图像"); return
        h, w = bin_img.shape[:2]
        pw = max(self._bin_preview_label.width() - 10, 280)
        scale = min(pw / max(w, 1), 160.0 / max(h, 1), 2.0)
        nw = max(int(w * scale), 1); nh = max(int(h * scale), 1)
        try:
            resized = cv2.resize(bin_img, (nw, nh), interpolation=cv2.INTER_NEAREST)
        except cv2.error:
            return
        qimg = mat_to_qimage(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))
        self._bin_preview_label.setPixmap(QPixmap.fromImage(qimg))
        self._bin_preview_label.setText("")

    # ==================== Save chars ====================

    def _on_save_chars(self):
        if not self._template_set or self._img_gray is None:
            QMessageBox.warning(self, "提示", "请先「设为基准模板」"); return
        if self._detect_roi is None: return
        self._read_params()
        save_dir = QFileDialog.getExistingDirectory(self, "选择字符模板保存目录")
        if not save_dir: return
        self._save_dir = save_dir
        rx, ry, rw, rh = self._detect_roi.get_roi()
        h_img, w_img = self._img_gray.shape[:2]
        rx = max(0, min(rx, w_img - 5)); ry = max(0, min(ry, h_img - 5))
        rw = min(rw, w_img - rx); rh = min(rh, h_img - ry)
        if rw < 10 or rh < 10:
            QMessageBox.warning(self, "错误", "检测框太小"); return
        roi_gray = self._img_gray[ry:ry + rh, rx:rx + rw].copy()
        char_rects, bin_img = detect_characters(roi_gray, self._params)
        if len(char_rects) == 0:
            QMessageBox.warning(self, "提示", "未检测到任何字符，请调整分割参数"); return
        trimmed = [trim_to_foreground(bin_img, cr, self._params['invert']) for cr in char_rects]
        rc = cluster_into_rows(trimmed)
        self._char_templates = []
        saved = 0
        for item in rc:
            cx, cy, cw, ch, row, col = item
            char_img = bin_img[cy:cy + ch, cx:cx + cw]
            if char_img.size == 0 or char_img.shape[0] < 2 or char_img.shape[1] < 2:
                continue
            fname = f"tpl_{row}_{col}.png"
            ok_written = cv2.imwrite(os.path.join(save_dir, fname), char_img)
            if not ok_written:
                self._status_label.setText("保存失败：无法写入文件")
                self._status_label.setStyleSheet(
                    "background:rgba(128,0,0,180); color:#ff4444; font-size:20px; font-weight:bold; padding:6px 16px; border-radius:6px;")
                QMessageBox.warning(self, "保存失败", f"无法写入文件：{os.path.join(save_dir, fname)}\n请检查路径权限或磁盘空间")
                return
            self._char_templates.append({
                'mat': char_img.copy(), 'x': rx + cx, 'y': ry + cy,
                'w': cw, 'h': ch, 'name': fname, 'row': row, 'col': col,
            })
            saved += 1
        self._status_label.setText(f"已保存 {saved} 个字符模板")
        self._status_label.setStyleSheet(
            "background:rgba(0,0,0,160); color:#00ff88; font-size:20px; font-weight:bold; padding:6px 16px; border-radius:6px;")
        self._info.setText(f"已保存 {saved} 个字符模板到：\n{save_dir}\n现在可以点击「切换下一张」进行检测")
        self._update_ui_state()

    # ==================== Phase B: Detection ====================

    def _on_next_image(self):
        if not self._template_set:
            QMessageBox.warning(self, "提示", "请先建立基准模板"); return
        if len(self._char_templates) == 0:
            QMessageBox.warning(self, "提示", "请先保存字符模板"); return
        path, _ = QFileDialog.getOpenFileName(self, "选择检测图片", "",
            "图片 (*.png *.jpg *.jpeg *.bmp *.tiff);;所有文件 (*.*)")
        if not path: return
        img = self._imread_unicode(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取：{path}"); return
        self._img_bgr = img.copy()
        self._img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._run_detection(path)

    def _run_detection(self, path=""):
        self._read_params()
        thresh = self._params['match_thresh'] / 100.0
        ax, ay, aw, ah = self._align_ref_rect
        dx, dy = self._detect_offset
        det_rx, det_ry, drw_ref, drh_ref = self._detect_ref_rect
        drw, drh = drw_ref, drh_ref
        h_img, w_img = self._img_gray.shape[:2]

        # --- 区域ROI限制：定位只在区域框内进行，减少全图搜索耗时 ---
        use_region = (self._region_ref_rect is not None and
                      self._cb_region_roi.isChecked())
        if use_region:
            rr_x, rr_y, rr_w, rr_h = self._region_ref_rect
            rr_x = max(0, rr_x); rr_y = max(0, rr_y)
            rr_w = min(rr_w, w_img - rr_x); rr_h = min(rr_h, h_img - rr_y)
        else:
            rr_x = rr_y = 0
            rr_w, rr_h = w_img, h_img

        # ================================================================
        # Step 1: ORB 特征匹配（主定位 + 旋转检测）
        #         当启用区域ROI时，只在区域框内进行匹配
        # ================================================================
        if use_region:
            # 区域ROI模式下：只取区域子图做模板匹配，减少计算量
            region_src = self._img_gray[rr_y:rr_y + rr_h, rr_x:rr_x + rr_w]
            _rx, _ry, _score = align_by_template_match(
                region_src, self._tpl_gray, (ax, ay, aw, ah))
            if _rx is not None:
                best_x = _rx + rr_x
                best_y = _ry + rr_y
                tmpl_score = _score
            else:
                best_x = best_y = None
                tmpl_score = 0.0
        else:
            best_x, best_y, tmpl_score = align_by_template_match(
                self._img_gray, self._tpl_gray, (ax, ay, aw, ah))


        if best_x is not None and tmpl_score >= 0.15:
            margin_x = max(int(aw * 0.3), 10)
            margin_y = max(int(ah * 0.3), 10)
            sx = max(0, best_x - margin_x)
            sy = max(0, best_y - margin_y)
            sw = min(w_img - sx, int(aw * 1.6))
            sh = min(h_img - sy, int(ah * 1.6))
            if use_region:
                sx = max(sx, rr_x); sy = max(sy, rr_y)
                sw = min(sw, rr_x + rr_w - sx)
                sh = min(sh, rr_y + rr_h - sy)
            orb_cx, orb_cy, orb_angle, orb_score, M_orb = align_by_feature_matching(
                self._img_gray, self._tpl_gray, (ax, ay, aw, ah),
                search_rect=(sx, sy, sw, sh))
        else:
            if use_region:
                orb_cx, orb_cy, orb_angle, orb_score, M_orb = align_by_feature_matching(
                    self._img_gray, self._tpl_gray, (ax, ay, aw, ah),
                    search_rect=(rr_x, rr_y, rr_w, rr_h))
            else:
                orb_cx, orb_cy, orb_angle, orb_score, M_orb = align_by_feature_matching(
                    self._img_gray, self._tpl_gray, (ax, ay, ah))

        # ================================================================
        # Step 2: 决定使用 ORB 对齐 还是 标准模板匹配
        # ================================================================
        use_orb_align = (M_orb is not None and orb_score >= 0.20)

        if use_orb_align:
            # ---- 2a: ORB 对齐路径（处理旋转） ----
            rotation_angle = orb_angle
            align_score = max(tmpl_score if best_x is not None else 0.0, orb_score)

            # 用仿射逆矩阵将检测图对齐到模板坐标系
            M_inv = cv2.invertAffineTransform(M_orb)
            img_aligned = cv2.warpAffine(
                self._img_gray, M_inv, (w_img, h_img),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)

            # 在对齐后的图像中，直接使用模板坐标！
            align_x, align_y = ax, ay
            align_cx = ax + aw / 2.0
            align_cy = ay + ah / 2.0

            # 检测框 ROI：直接用模板坐标裁剪对齐图
            _drx = det_rx
            _dry = det_ry
            _rw = min(drw, w_img - _drx)
            _rh = min(drh, h_img - _dry)
            if _rw < 10 or _rh < 10:
                self._info.setText("ROI联动后检测框太小"); return
            roi_gray = img_aligned[_dry:_dry + _rh, _drx:_drx + _rw].copy()
            roi_w, roi_h = _rw, _rh
            M_d2s = M_orb

            def clamp_rect(cx, cy, cw, ch):
                cx = max(0, min(cx, roi_w - 1))
                cy = max(0, min(cy, roi_h - 1))
                return cx, cy, max(1, min(cw, roi_w - cx)), max(1, min(ch, roi_h - cy))

            def roi_to_src(rx, ry, rw2, rh2):
                mcx = _drx + rx + rw2 / 2.0
                mcy = _dry + ry + rh2 / 2.0
                src_cx = M_d2s[0, 0] * mcx + M_d2s[0, 1] * mcy + M_d2s[0, 2]
                src_cy = M_d2s[1, 0] * mcx + M_d2s[1, 1] * mcy + M_d2s[1, 2]
                return src_cx, src_cy, rw2, rh2, rotation_angle

        elif best_x is not None and tmpl_score >= 0.25:
            # ---- 2b: 标准模板匹配路径（无旋转） ----
            rotation_angle = 0.0
            align_score = tmpl_score
            align_x, align_y = best_x, best_y
            align_cx = best_x + aw / 2.0
            align_cy = best_y + ah / 2.0

            _drx = align_x + dx
            _dry = align_y + dy
            _drx = max(0, min(_drx, w_img - 20))
            _dry = max(0, min(_dry, h_img - 20))
            _rw = min(drw, w_img - _drx)
            _rh = min(drh, h_img - _dry)
            if _rw < 10 or _rh < 10:
                self._info.setText("ROI联动后检测框太小"); return
            roi_gray = self._img_gray[_dry:_dry + _rh, _drx:_drx + _rw].copy()
            roi_w, roi_h = _rw, _rh
            M_d2s = None

            def clamp_rect(cx, cy, cw, ch):
                cx = max(0, min(cx, roi_w - 1))
                cy = max(0, min(cy, roi_h - 1))
                return cx, cy, max(1, min(cw, roi_w - cx)), max(1, min(ch, roi_h - cy))

            def roi_to_src(rx, ry, rw2, rh2):
                return float(_drx + rx), float(_dry + ry), float(rw2), float(rh2), 0.0
        else:
            self._show_image(self._img_bgr.copy(), [], show_rois=False)
            self._status_label.setText("对齐失败")
            self._status_label.setStyleSheet(
                "background:rgba(128,0,0,180); color:#ff4444; font-size:22px; font-weight:bold; "
                "padding:8px 20px; border-radius:6px;")
            reason = ("ORB=%.2f" % orb_score) if orb_cx is not None else "ORB未匹配"
            self._info.setText(
                "对齐失败（标准=%.2f, %s）\n图像可能变形严重或纹理不足" % (tmpl_score, reason))
            self._bin_preview_label.setText("对齐失败"); return

        is_rotated = abs(rotation_angle) > 1.0

        # ================================================================
        # Step 3: 字符切分（在轴对齐的 roi_gray 上执行）
        # ================================================================
        char_rects, bin_img = detect_characters(roi_gray, self._params)

        char_rects_clamped = []
        for cr in char_rects:
            clamped = clamp_rect(cr[0], cr[1], cr[2], cr[3])
            if clamped[2] >= 2 and clamped[3] >= 2:
                char_rects_clamped.append(clamped)

        trimmed = [trim_to_foreground(bin_img, cr, self._params['invert'])
                   for cr in char_rects_clamped]
        trimmed_clamped = []
        for cr in trimmed:
            clamped = clamp_rect(cr[0], cr[1], cr[2], cr[3])
            if clamped[2] >= 2 and clamped[3] >= 2:
                trimmed_clamped.append(clamped)
        trimmed = trimmed_clamped

        self._update_bin_preview(bin_img)

        # ================================================================
        # Step 4: 绘制定位框与检测框（始终在原图上）
        # ================================================================
        display = self._img_bgr.copy()
        # 关键修复：只要 M_d2s 存在（ORB路径），必须用仿射变换绘制，
        #         不能因为角度小就走 else 分支（else 分支的坐标是模板坐标，
        #         在原图上画会偏移到错误位置）
        if M_d2s is not None:
            align_corners_src = _transform_rect(M_d2s, ax, ay, aw, ah)
            cv2.polylines(display, [np.int32(align_corners_src)], True,
                          (255, 120, 60), 2, lineType=cv2.LINE_AA)
            det_corners_src = _transform_rect(M_d2s, det_rx, det_ry, drw, drh)
            cv2.polylines(display, [np.int32(det_corners_src)], True,
                          (40, 200, 255), 2, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(display, (align_x, align_y), (align_x + aw, align_y + ah),
                          (255, 120, 60), 2)
            cv2.rectangle(display, (_drx, _dry), (_drx + roi_w, _dry + roi_h),
                          (40, 200, 255), 2)

        # 区域ROI框显示：启用时在原图上绘制紫色区域框
        if use_region:
            if M_d2s is not None:
                region_corners = _transform_rect(M_d2s, rr_x, rr_y, rr_w, rr_h)
                cv2.polylines(display, [np.int32(region_corners)], True,
                              (160, 80, 220), 2, lineType=cv2.LINE_AA)
            else:
                # 标准匹配路径：区域框跟随定位偏移
                region_det_x = align_x + (rr_x - ax)
                region_det_y = align_y + (rr_y - ay)
                cv2.rectangle(display, (int(region_det_x), int(region_det_y)),
                              (int(region_det_x + rr_w), int(region_det_y + rr_h)),
                              (160, 80, 220), 2)

        if len(trimmed) == 0:
            self._show_image(display, [], show_rois=False)
            self._status_label.setText("NG")
            self._status_label.setStyleSheet(
                "background:rgba(128,0,0,180); color:#ff4444; font-size:28px; font-weight:bold; "
                "padding:8px 24px; border-radius:8px;")
            self._info.setText("检测框内未找到任何字符"); return

        # ================================================================
        # Step 5: key-based 模板匹配
        # ================================================================
        detect_rc = cluster_into_rows(trimmed)
        tmpl_map = {(t['row'], t['col']): t['mat'] for t in self._char_templates}
        ok_list, ng_list = [], []

        for item in detect_rc:
            cx, cy, cw, ch, row, col = item
            cx, cy, cw, ch = clamp_rect(cx, cy, cw, ch)
            char_sub = bin_img[cy:cy + ch, cx:cx + cw]
            label = "%d_%d" % (row, col)
            if char_sub.size == 0:
                ng_list.append((cx, cy, cw, ch, False, label + "空", 0.0))
                continue
            key = (row, col)
            tpl = tmpl_map.get(key)
            match_score = match_template_score(tpl, char_sub) if tpl is not None else 0.0
            if match_score >= thresh:
                ok_list.append((cx, cy, cw, ch, True, label + "%.0f" % (match_score * 100) + "%", match_score))
            else:
                ng_list.append((cx, cy, cw, ch, False, label + "%.0f" % (match_score * 100) + "%", match_score))

        # ================================================================
        # Step 6: 坐标映射回原图 + 绘制结果框
        # ================================================================
        for kind, lst, box_color, ok_flag in [
            ("ok", ok_list, (0, 200, 0), True),
            ("ng", ng_list, (0, 0, 200), False)]:
            for item in lst:
                cx, cy, cw, ch, _, label, _ = item[:7]
                src_cx, src_cy, sw, sh, s_angle = roi_to_src(cx, cy, cw, ch)
                half_w, half_h = sw / 2.0, sh / 2.0
                x1 = src_cx - half_w
                y1 = src_cy - half_h
                x2 = src_cx + half_w
                y2 = src_cy + half_h
                if abs(s_angle) > 1.0:
                    _draw_rotated_box(display, (src_cx, src_cy), (sw, sh), s_angle,
                                      box_color, ok_flag, label)
                else:
                    fill_color = (0, 120, 0) if ok_flag else (120, 0, 0)
                    overlay = display.copy()
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                                  fill_color, -1)
                    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
                    cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)),
                                  box_color, 2, cv2.LINE_AA)
                    if label:
                        cv2.putText(display, label, (int(x1) + 2, int(y1) - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # ================================================================
        # Step 7: 结果判定与状态显示
        # ================================================================
        is_pass = len(ng_list) == 0
        self._show_image(display, [], show_rois=False)

        if is_pass:
            self._status_label.setText("OK")
            self._status_label.setStyleSheet(
                "background:rgba(0,128,0,180); color:#00ff88; font-size:32px; font-weight:bold; "
                "padding:8px 24px; border-radius:8px;")
        else:
            self._status_label.setText("NG")
            self._status_label.setStyleSheet(
                "background:rgba(128,0,0,180); color:#ff4444; font-size:32px; font-weight:bold; "
                "padding:8px 24px; border-radius:8px;")

        n_char = len(detect_rc)
        best_match = max((o[6] for o in ok_list), default=0.0)
        worst_match = min((n[6] for n in ng_list), default=0.0)
        import os
        info_parts = [
            "检测：" + os.path.basename(path) if path else "",
            "标准匹配：%.2f" % tmpl_score,
        ]
        if orb_cx is not None:
            info_parts.append("ORB特征匹配：%.2f  角度：%.1f" % (orb_score, orb_angle))
        info_parts.extend([
            "综合对齐得分：%.2f" % align_score,
            "OK=%d  NG=%d  总计=%d" % (len(ok_list), len(ng_list), n_char),
            "匹配阈值：%d%%" % self._params['match_thresh'],
        ])
        if ok_list:
            info_parts.append("最佳匹配：%.1f" % best_match)
        if ng_list:
            info_parts.append("最差：%.1f" % worst_match)
        self._info.setText("\n".join(info_parts))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


# ========================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
