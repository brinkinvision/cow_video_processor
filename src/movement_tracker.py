"""
Класс для отслеживания движения коровы по трем критериям
"""

import math
import time
from typing import Optional, Dict, List, Tuple

from config import (
    SPINE_INDICES,
    LEG_POINTS_MAP,
    NECK_INDEX,
    END_INDEX,
    MOVEMENT_CONFIG,
    THRESHOLD_CONFIG,
)


class CowMovementTracker:
    """Класс для отслеживания движения коровы по трем критериям"""
    
    def __init__(self, track_id: int):
        """
        Инициализация трекера движения коровы
        
        Args:
            track_id: Уникальный ID коровы
        """
        self.track_id = track_id
        self.last_seen = None
        
        # История для сглаживания решения о движении
        self.movement_history: List[bool] = []
        self.max_history_length = MOVEMENT_CONFIG["max_history_length"]
        self.movement_threshold = MOVEMENT_CONFIG["movement_threshold"]
        
        # Критерий 1: Движение спины
        self._initial_spine_positions = None
        self._last_average_distance_px = 0.0
        
        # Критерий 2: Движение ног
        self.leg_history: Dict[str, List[Tuple[float, float, float]]] = {}
        self.keypoints_history: List = []
        
        # Критерий 3: Движение bbox
        self._initial_bbox_center = None
        self._last_bbox_distance_from_start = 0.0
        self._increasing_movement = False
        
        # Результат
        self.is_moving_flag = False
    
    def update(
        self,
        keypoints,
        bbox,
        threshold_relative: Optional[float] = None,
        threshold_stable: Optional[float] = None,
        verbose: bool = False
    ) -> bool:
        """
        Обновляет состояние движения на основе текущего кадра
        
        Args:
            keypoints: массив кейпоинтов (N, 3) где N=14, 3=(x, y, confidence)
            bbox: [x1, y1, x2, y2] bounding box
            threshold_relative: порог для движения ног (доля длины тела)
            threshold_stable: порог для движения спины
            verbose: выводить отладочную информацию
            
        Returns:
            bool: True если корова движется, False если стоит
        """
        # Используем значения по умолчанию из конфига
        if threshold_relative is None:
            threshold_relative = THRESHOLD_CONFIG["threshold_relative"]
        if threshold_stable is None:
            threshold_stable = THRESHOLD_CONFIG["threshold_stable"]
        
        if keypoints is None or len(keypoints) == 0:
            self._update_smoothed_movement(False)
            return False
        
        self.last_seen = time.time()
        
        # Сохраняем keypoints в историю для расчета body_length
        self.keypoints_history.append(keypoints)
        if len(self.keypoints_history) > 10:
            self.keypoints_history.pop(0)
        
        # === КРИТЕРИЙ 1: Движение спины ===
        spine_moving = self._check_spine_movement(keypoints, threshold_stable)
        
        # === КРИТЕРИЙ 2: Движение хотя бы одной ноги ===
        leg_moving = self._check_leg_movement(keypoints, threshold_relative)
        
        # === КРИТЕРИЙ 3: Удаление bbox от начальной точки ===
        bbox_moving = self._check_bbox_movement(bbox)
        
        if verbose:
            print(f"[ID {self.track_id}] Spine: {spine_moving}, Leg: {leg_moving}, BBox: {bbox_moving}")
        
        # === Финальное решение ===
        raw_moving = spine_moving and leg_moving and bbox_moving
        
        # Сглаживание по времени
        self._update_smoothed_movement(raw_moving)
        self.is_moving_flag = self._get_smoothed_movement()
        
        if verbose:
            print(f"[ID {self.track_id}] Raw: {raw_moving}, Smoothed: {self.is_moving_flag}, History: {self.movement_history}")
        
        return self.is_moving_flag
    
    def _check_spine_movement(self, keypoints, threshold_stable: float) -> bool:
        """Проверяет движение спины (критерий 1)"""
        confidence_threshold = MOVEMENT_CONFIG["keypoint_confidence_threshold"]
        
        current_points = []
        for idx in SPINE_INDICES:
            if idx < len(keypoints) and keypoints[idx][2] > confidence_threshold:
                current_points.append((keypoints[idx][0], keypoints[idx][1]))
        
        if len(current_points) < 3:  # Недостаточно видимых точек
            return False
        
        # Инициализация начальных позиций
        if self._initial_spine_positions is None:
            self._initial_spine_positions = current_points
            self._last_average_distance_px = 0.0
            return False
        
        # Если количество точек изменилось - пересоздаем базовую позицию
        if len(current_points) != len(self._initial_spine_positions):
            self._initial_spine_positions = current_points
            self._last_average_distance_px = 0.0
            return False
        
        distances = [
            math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            for p1, p2 in zip(self._initial_spine_positions, current_points)
        ]
        average_distance_px = sum(distances) / len(distances)
        
        # Нормализация по длине тела
        body_length = self._get_body_length(keypoints)
        if not body_length or body_length == 0:
            relative_distance = 0.0
        else:
            relative_distance = average_distance_px / body_length
        
        # Проверяем увеличение смещения
        spine_moving = False
        if relative_distance > self._last_average_distance_px + threshold_stable:
            spine_moving = True
        elif relative_distance > self._last_average_distance_px:
            spine_moving = True
        
        self._last_average_distance_px = relative_distance
        
        return spine_moving
    
    def _check_leg_movement(self, keypoints, threshold_relative: float) -> bool:
        """Проверяет движение ног (критерий 2)"""
        history_length = MOVEMENT_CONFIG["leg_history_length"]
        confidence_threshold = MOVEMENT_CONFIG["keypoint_confidence_threshold"]
        
        body_length = self._get_body_length(keypoints)
        if not body_length or body_length == 0:
            return False
        
        current_time = time.time()
        
        # Сохраняем текущие позиции ног в историю
        for leg_name, indices in LEG_POINTS_MAP.items():
            if leg_name not in self.leg_history:
                self.leg_history[leg_name] = []
            
            # Собираем точки этой ноги
            leg_points = []
            for idx in indices:
                if idx < len(keypoints) and keypoints[idx][2] > confidence_threshold:
                    leg_points.append((keypoints[idx][0], keypoints[idx][1], current_time))
            
            if len(leg_points) == 2:
                # Добавляем среднюю позицию ноги
                avg_x = (leg_points[0][0] + leg_points[1][0]) / 2
                avg_y = (leg_points[0][1] + leg_points[1][1]) / 2
                self.leg_history[leg_name].append((avg_x, avg_y, current_time))
                
                # Ограничиваем историю
                if len(self.leg_history[leg_name]) > history_length + 1:
                    self.leg_history[leg_name].pop(0)
        
        # Проверяем движение хотя бы одной ноги
        any_leg_moving = False
        for leg_name, history in self.leg_history.items():
            if len(history) < 2:
                continue
            
            # Вычисляем среднее смещение за последние кадры
            total_distance = 0.0
            count = 0
            for i in range(len(history) - 1):
                dx = history[i+1][0] - history[i][0]
                dy = history[i+1][1] - history[i][1]
                distance = math.hypot(dx, dy)
                total_distance += distance
                count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                relative_distance = avg_distance / body_length
                
                if relative_distance > threshold_relative:
                    any_leg_moving = True
                    break
        
        return any_leg_moving
    
    def _check_bbox_movement(self, bbox) -> bool:
        """Проверяет удаление bbox от начальной точки (критерий 3)"""
        if bbox is None or len(bbox) < 4:
            return False
        
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Инициализация начальной позиции
        if self._initial_bbox_center is None:
            self._initial_bbox_center = (bbox_center_x, bbox_center_y)
            self._last_bbox_distance_from_start = 0.0
            return False
        
        # Вычисляем расстояние от начальной позиции
        dx = bbox_center_x - self._initial_bbox_center[0]
        dy = bbox_center_y - self._initial_bbox_center[1]
        distance = math.hypot(dx, dy)
        
        # Проверяем увеличение расстояния
        if distance > self._last_bbox_distance_from_start:
            self._increasing_movement = True
        else:
            self._increasing_movement = False
        
        self._last_bbox_distance_from_start = distance
        
        return self._increasing_movement
    
    def _get_body_length(self, keypoints) -> Optional[float]:
        """
        Вычисляет длину тела как расстояние между neck и end
        
        Returns:
            float or None: длина тела в пикселях или None если точки не видны
        """
        confidence_threshold = MOVEMENT_CONFIG["keypoint_confidence_threshold"]
        
        if (NECK_INDEX < len(keypoints) and END_INDEX < len(keypoints) and
            keypoints[NECK_INDEX][2] > confidence_threshold and 
            keypoints[END_INDEX][2] > confidence_threshold):
            dx = keypoints[NECK_INDEX][0] - keypoints[END_INDEX][0]
            dy = keypoints[NECK_INDEX][1] - keypoints[END_INDEX][1]
            return math.hypot(dx, dy)
        return None
    
    def _update_smoothed_movement(self, current_decision: bool):
        """Добавляет текущее решение в историю"""
        self.movement_history.append(current_decision)
        if len(self.movement_history) > self.max_history_length:
            self.movement_history.pop(0)
    
    def _get_smoothed_movement(self) -> bool:
        """Возвращает сглаженное решение на основе истории"""
        if not self.movement_history:
            return False
        moving_ratio = sum(self.movement_history) / len(self.movement_history)
        return moving_ratio >= self.movement_threshold
    
    def is_active(self, timeout_seconds: Optional[int] = None) -> bool:
        """
        Проверяет, активна ли корова (видели недавно)
        
        Args:
            timeout_seconds: таймаут в секундах
            
        Returns:
            bool: True если корова активна
        """
        if timeout_seconds is None:
            timeout_seconds = MOVEMENT_CONFIG["timeout_seconds"]
            
        if self.last_seen is None:
            return False
        return (time.time() - self.last_seen) < timeout_seconds
    
    def get_leg_movement_status(self) -> Dict[str, bool]:
        """
        Возвращает статус движения каждой ноги для визуализации
        
        Returns:
            dict: {leg_name: is_moving}
        """
        threshold_relative = THRESHOLD_CONFIG["threshold_relative"]
        history_length = MOVEMENT_CONFIG["leg_history_length"]
        
        # Получаем длину тела из последних keypoints
        body_length = None
        if self.keypoints_history:
            last_keypoints = self.keypoints_history[-1]
            body_length = self._get_body_length(last_keypoints)
        
        # Если нет длины тела, не можем определить движение
        if not body_length or body_length == 0:
            return {leg: False for leg in LEG_POINTS_MAP.keys()}
        
        leg_status = {}
        
        # Проверяем каждую ногу
        for leg_name in LEG_POINTS_MAP.keys():
            if leg_name not in self.leg_history or len(self.leg_history[leg_name]) < 2:
                leg_status[leg_name] = False
                continue
            
            history = self.leg_history[leg_name]
            
            # Вычисляем среднее смещение за последние кадры
            total_distance = 0.0
            count = 0
            
            start_idx = max(0, len(history) - history_length - 1)
            for i in range(start_idx, len(history) - 1):
                dx = history[i+1][0] - history[i][0]
                dy = history[i+1][1] - history[i][1]
                distance = math.hypot(dx, dy)
                total_distance += distance
                count += 1
            
            if count > 0:
                avg_distance = total_distance / count
                relative_distance = avg_distance / body_length
                leg_status[leg_name] = relative_distance > threshold_relative
            else:
                leg_status[leg_name] = False
        
        return leg_status
