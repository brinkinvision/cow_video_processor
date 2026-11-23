"""
Класс для обработки видео с использованием YOLO Pose модели и ByteTracker
"""

import cv2
import time
from pathlib import Path
from typing import Optional, Dict
from ultralytics import YOLO

from .movement_tracker import CowMovementTracker
from config import YOLO_CONFIG, VIDEO_CONFIG, MOVEMENT_CONFIG, ACTIVATION_CONFIG
from utils import (
    draw_keypoints_only,
    draw_leg_lines,
    draw_bbox_with_label,
    draw_frame_info,
)


class VideoProcessor:
    """
    Класс для обработки видео с детекцией и отслеживанием коров
    """
    
    def __init__(self, model_path: str, conf_threshold: Optional[float] = None):
        """
        Инициализация процессора видео
        
        Args:
            model_path: Путь к обученной модели YOLO
            conf_threshold: Порог уверенности для детекций
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold or YOLO_CONFIG["conf_threshold"]
        self.tracker_name = YOLO_CONFIG["tracker"]
        
        print(f"Загрузка модели из {model_path}...")
        self.model = YOLO(model_path)
        
        # Словарь для хранения трекеров движения коров
        self.movement_trackers: Dict[int, CowMovementTracker] = {}
        
        # Словарь для отслеживания активации коров (появилась ли точка neck)
        self.activated_cows: Dict[int, bool] = {}
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Обрабатывает видео с использованием YOLO Pose модели и ByteTracker
        
        Args:
            video_path: Путь к входному видео
            output_path: Путь для сохранения выходного видео (опционально)
            verbose: Включить отладочный вывод для каждой коровы
        """
        # Открываем видео
        print(f"Открытие видео {video_path}...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Параметры видео: {width}x{height}, {fps} FPS, {total_frames} кадров")
        
        # Создаем VideoWriter для сохранения результата
        if output_path is None:
            # Генерируем имя выходного файла на основе входного
            video_path_obj = Path(video_path)
            video_dir = video_path_obj.parent
            video_stem = video_path_obj.stem
            video_ext = video_path_obj.suffix or ".mp4"
            
            # Если расширение не .mp4, используем .mp4
            if video_ext.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
                video_ext = '.mp4'
            
            # Создаем папку visualized если её нет
            output_dir = video_dir / "visualized"
            output_dir.mkdir(exist_ok=True)
            
            # Базовое имя выходного файла в папке visualized
            base_output = output_dir / f"{video_stem}_processed{video_ext}"
            
            # Если файл существует, добавляем номер
            output_path = base_output
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{video_stem}_processed_{counter}{video_ext}"
                counter += 1
            
            output_path = str(output_path)
        else:
            # Если путь задан явно, используем его как есть
            # но также создаем родительские папки если их нет
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["fourcc"])
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Начало обработки видео...")
        print(f"Результат будет сохранен в: {output_path}")
        
        frame_count = 0
        
        # Переменные для расчета FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        fps_update_interval = 10  # обновляем FPS каждые 10 кадров
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                fps_frame_count += 1
                
                # Вычисляем реальный FPS
                if fps_frame_count >= fps_update_interval:
                    elapsed_time = time.time() - fps_start_time
                    if elapsed_time > 0:
                        current_fps = fps_frame_count / elapsed_time
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Обрабатываем кадр
                self._process_frame(frame, verbose)
                
                # Добавляем информацию на кадр
                draw_frame_info(frame, frame_count, total_frames, current_fps)
                
                # Записываем кадр
                out.write(frame)
                
                # Показываем прогресс
                if frame_count % VIDEO_CONFIG["progress_update_interval"] == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Обработано: {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {current_fps:.1f}")
        
        finally:
            # Освобождаем ресурсы
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"\nОбработка завершена!")
        print(f"Обработано {frame_count} кадров")
        print(f"Результат сохранен в: {output_path}")
    
    def _process_frame(self, frame, verbose: bool = False):
        """
        Обрабатывает один кадр видео
        
        Args:
            frame: Кадр для обработки
            verbose: Включить отладочный вывод
        """
        # Запускаем трекинг с ByteTracker
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            persist=True,
            tracker=self.tracker_name,
            verbose=False
        )
        
        # Обрабатываем результаты
        if results and len(results) > 0:
            result = results[0]
            
            # Проверяем наличие боксов
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                # Получаем track IDs если доступны
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = [-1] * len(boxes)
                
                # Получаем keypoints если доступны
                keypoints_data = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                
                # Обрабатываем каждую корову
                for idx, (box, conf, track_id) in enumerate(zip(boxes, confs, track_ids)):
                    self._process_cow(
                        frame, box, conf, track_id, 
                        keypoints_data[idx] if keypoints_data is not None and idx < len(keypoints_data) else None,
                        verbose
                    )
        
        # Удаляем неактивных коров из трекеров
        self._cleanup_inactive_trackers()
    
    def _process_cow(
        self,
        frame,
        box,
        conf: float,
        track_id: int,
        keypoints,
        verbose: bool = False
    ):
        """
        Обрабатывает одну корову на кадре
        
        Args:
            frame: Кадр для рисования
            box: Bounding box [x1, y1, x2, y2]
            conf: Уверенность детекции
            track_id: ID трека
            keypoints: Массив keypoints (N, 3)
            verbose: Включить отладочный вывод
        """
        # Пропускаем коров без track_id
        if track_id == -1:
            return
        
        # Проверяем активацию коровы (появились ли ноги)
        if not self._is_cow_activated(track_id, keypoints, verbose):
            if verbose:
                print(f"[ID {track_id}] Корова еще не активирована (ждем появления ног)")
            return
        
        # Инициализируем трекер для новых активированных коров
        if track_id not in self.movement_trackers:
            self.movement_trackers[track_id] = CowMovementTracker(track_id)
            print(f"[INFO] Создан новый трекер для коровы ID: {track_id}")
        
        # Определяем движение коровы
        is_moving = False
        if keypoints is not None:
            is_moving = self.movement_trackers[track_id].update(keypoints, box, verbose=verbose)
        
        # Рисуем bounding box с меткой
        draw_bbox_with_label(frame, box, conf, track_id, is_moving)
        
        # Рисуем keypoints
        if keypoints is not None:
            draw_keypoints_only(frame, keypoints)
            
            # Рисуем линии на ногах с индикацией движения
            if track_id in self.movement_trackers:
                leg_status = self.movement_trackers[track_id].get_leg_movement_status()
                draw_leg_lines(frame, keypoints, leg_status)
    
    def _is_cow_activated(self, track_id: int, keypoints, verbose: bool = False) -> bool:
        """
        Проверяет, активирована ли корова (появилась ли хотя бы одна нога)
        
        Args:
            track_id: ID трека коровы
            keypoints: Массив keypoints (N, 3)
            verbose: Включить отладочный вывод
            
        Returns:
            bool: True если корова активирована, False если еще нет
        """
        # Если проверка отключена, активируем все коровы сразу
        if not ACTIVATION_CONFIG["enable_leg_check"]:
            self.activated_cows[track_id] = True
            return True
        
        # Если корова уже была активирована, возвращаем True
        if track_id in self.activated_cows and self.activated_cows[track_id]:
            return True
        
        # Проверяем наличие keypoints
        if keypoints is None or len(keypoints) == 0:
            return False
        
        # Индексы точек ног (8 точек: 4 ноги x 2 точки)
        # left_forth: 6, 7
        # right_forth: 8, 9
        # left_back: 10, 11
        # right_back: 12, 13
        leg_indices = [6, 7, 8, 9, 10, 11, 12, 13]
        
        # Проверяем каждую точку ноги
        visible_legs = []
        leg_info = {}
        
        for idx in leg_indices:
            if idx < len(keypoints):
                leg_kp = keypoints[idx]
                confidence = leg_kp[2]
                
                if confidence >= ACTIVATION_CONFIG["leg_confidence_threshold"]:
                    visible_legs.append(idx)
                    leg_info[idx] = {
                        "x": leg_kp[0],
                        "y": leg_kp[1],
                        "conf": confidence
                    }
        
        # Определяем условие активации
        if ACTIVATION_CONFIG["require_all_legs"]:
            # Требуется видеть все 8 точек ног
            activated = len(visible_legs) >= 8
            condition = "все точки ног"
        else:
            # Достаточно хотя бы одной точки ноги
            activated = len(visible_legs) >= 1
            condition = "хотя бы одна точка ноги"
        
        if activated:
            self.activated_cows[track_id] = True
            
            # Выводим детальную информацию
            leg_names = {
                6: "left_front_tp", 7: "left_front_bt",
                8: "right_front_tp", 9: "right_front_bt",
                10: "left_back_tp", 11: "left_back_bt",
                12: "right_back_tp", 13: "right_back_bt"
            }
            
            visible_names = [leg_names[idx] for idx in visible_legs]
            
            print(f"[INFO] Корова ID {track_id} активирована! ({condition})")
            print(f"       Видимые точки ног ({len(visible_legs)}): {', '.join(visible_names)}")
            
            if verbose and leg_info:
                print(f"       Детали:")
                for idx, info in leg_info.items():
                    print(f"         {leg_names[idx]}: x={info['x']:.1f}, y={info['y']:.1f}, conf={info['conf']:.2f}")
            
            return True
        
        if verbose:
            print(f"[ID {track_id}] Недостаточно точек ног для активации: {len(visible_legs)}/8")
        
        return False
    
    def _cleanup_inactive_trackers(self):
        """Удаляет неактивных коров из трекеров"""
        timeout_seconds = MOVEMENT_CONFIG["timeout_seconds"]
        to_remove = []
        
        for track_id, tracker in self.movement_trackers.items():
            if not tracker.is_active(timeout_seconds=timeout_seconds):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            print(f"[INFO] Удален трекер для коровы ID: {track_id} (не видели > {timeout_seconds} сек)")
            del self.movement_trackers[track_id]
            # Также удаляем запись об активации
            if track_id in self.activated_cows:
                del self.activated_cows[track_id]
