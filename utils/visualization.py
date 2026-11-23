"""
Утилиты для визуализации результатов обработки видео
"""

import cv2
from typing import Dict

from config import (
    KEYPOINT_COLORS,
    LEG_POINTS_MAP,
    LEG_SHORT_NAMES,
    VISUALIZATION_CONFIG,
    MOVEMENT_CONFIG,
)


def draw_keypoints_only(frame, keypoints):
    """
    Рисует только keypoints (без скелета) на изображении
    
    Args:
        frame: Изображение для рисования
        keypoints: Массив keypoints (N, 3) где N - количество точек, 3 - (x, y, confidence)
    """
    if keypoints is None or len(keypoints) == 0:
        return
    
    confidence_threshold = MOVEMENT_CONFIG["keypoint_confidence_threshold"]
    radius = VISUALIZATION_CONFIG["keypoint_radius"]
    border_thickness = VISUALIZATION_CONFIG["keypoint_border_thickness"]
    
    # Рисуем только keypoints (без линий скелета)
    for idx, kp in enumerate(keypoints):
        if kp[2] > confidence_threshold:
            x, y = int(kp[0]), int(kp[1])
            color = KEYPOINT_COLORS[idx] if idx < len(KEYPOINT_COLORS) else (255, 255, 255)
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), border_thickness)


def draw_leg_lines(frame, keypoints, leg_status: Dict[str, bool]):
    """
    Рисует линии на ногах коровы с цветовой индикацией движения
    
    Args:
        frame: Изображение для рисования
        keypoints: Массив keypoints (N, 3) где N=14, 3=(x, y, confidence)
        leg_status: Словарь {leg_name: is_moving} со статусом движения каждой ноги
    """
    if keypoints is None or len(keypoints) == 0:
        return
    
    confidence_threshold = MOVEMENT_CONFIG["keypoint_confidence_threshold"]
    moving_color = VISUALIZATION_CONFIG["moving_color"]
    stopped_color = VISUALIZATION_CONFIG["stopped_color"]
    line_thickness = VISUALIZATION_CONFIG["line_thickness"]
    
    # Рисуем линии для каждой ноги
    for leg_name, (top_idx, bot_idx) in LEG_POINTS_MAP.items():
        # Проверяем, что индексы в пределах массива и точки видимы
        if (top_idx < len(keypoints) and bot_idx < len(keypoints) and
            keypoints[top_idx][2] > confidence_threshold and 
            keypoints[bot_idx][2] > confidence_threshold):
            
            # Координаты точек
            pt1 = (int(keypoints[top_idx][0]), int(keypoints[top_idx][1]))
            pt2 = (int(keypoints[bot_idx][0]), int(keypoints[bot_idx][1]))
            
            # Определяем цвет линии по статусу движения
            is_moving = leg_status.get(leg_name, False)
            line_color = moving_color if is_moving else stopped_color
            
            # Рисуем линию на кадре
            cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)
            
            # Добавляем подпись с названием ноги
            text_x = (pt1[0] + pt2[0]) // 2
            text_y = (pt1[1] + pt2[1]) // 2 - 10
            text_position = (text_x, text_y)
            
            # Рисуем короткое название (LF, RF, LB, RB)
            leg_short_name = LEG_SHORT_NAMES.get(leg_name, "")
            cv2.putText(
                frame,
                leg_short_name,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                line_color,
                2,
                cv2.LINE_AA
            )


def draw_bbox_with_label(frame, box, conf, track_id, is_moving: bool):
    """
    Рисует bounding box с меткой на кадре
    
    Args:
        frame: Изображение для рисования
        box: [x1, y1, x2, y2] координаты bbox
        conf: уверенность детекции
        track_id: ID трека
        is_moving: флаг движения
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Выбираем цвет bbox в зависимости от движения
    moving_color = VISUALIZATION_CONFIG["moving_color"]
    stopped_color = VISUALIZATION_CONFIG["stopped_color"]
    box_color = moving_color if is_moving else stopped_color
    
    # Рисуем bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # Формируем текст с ID, confidence и статусом движения
    status = "MOVE" if is_moving else "STOP"
    if track_id != -1:
        label = f"ID: {track_id} | {status} | Conf: {conf:.2f}"
    else:
        label = f"{status} | Conf: {conf:.2f}"
    
    # Рисуем текст сверху bbox
    text_scale = VISUALIZATION_CONFIG["text_scale"]
    text_thickness = VISUALIZATION_CONFIG["text_thickness"]
    
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    label_y = max(y1 - 10, label_size[1] + 5)
    
    # Фон для текста
    cv2.rectangle(
        frame,
        (x1, label_y - label_size[1] - 5),
        (x1 + label_size[0], label_y + 5),
        box_color,
        -1
    )
    
    # Текст
    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 0, 0),
        text_thickness
    )


def draw_frame_info(frame, frame_count: int, total_frames: int, fps: float):
    """
    Рисует информацию о кадре и FPS на изображении
    
    Args:
        frame: Изображение для рисования
        frame_count: номер текущего кадра
        total_frames: общее количество кадров
        fps: реальный FPS обработки
    """
    text_scale = VISUALIZATION_CONFIG["label_text_scale"]
    text_thickness = VISUALIZATION_CONFIG["text_thickness"]
    
    # Информация о прогрессе
    progress_text = f"Frame: {frame_count}/{total_frames}"
    cv2.putText(
        frame,
        progress_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (255, 255, 255),
        text_thickness
    )
    
    # Информация о FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (255, 255, 255),
        text_thickness
    )
