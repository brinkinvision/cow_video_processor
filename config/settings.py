"""
Конфигурационный файл с настройками для обработки видео
"""

# Определение названий ключевых точек для модели runs_1 (14 keypoints)
KEYPOINT_NAMES = [
    "head",            # 0
    "neck",            # 1
    "start",           # 2
    "middle",          # 3
    "hook",            # 4
    "end",             # 5
    "left_front_tp",   # 6
    "left_front_bt",   # 7
    "right_front_tp",  # 8
    "right_front_bt",  # 9
    "left_back_tp",    # 10
    "left_back_bt",    # 11
    "right_back_tp",   # 12
    "right_back_bt",   # 13
]

# Цвета для визуализации keypoints (BGR формат)
KEYPOINT_COLORS = [
    (255, 0, 255),    # head - розовый
    (255, 255, 0),    # neck - голубой
    (0, 255, 0),      # start - зеленый
    (255, 0, 0),      # middle_high - синий
    (0, 165, 255),    # 80 - оранжевый
    (0, 0, 255),      # end - красный
    (255, 255, 255),  # left_forth_top - белый
    (128, 128, 128),  # left_forth_bot - серый
    (255, 192, 203),  # right_forth_top - светло-розовый
    (147, 20, 255),   # right_forth_bot - темно-розовый
    (0, 255, 127),    # left_back_top - весенний зеленый
    (127, 255, 0),    # left_back_bot - зелено-желтый
    (255, 127, 0),    # right_back_top - оранжевый
    (127, 0, 255),    # right_back_bot - фиолетовый
]

# Индексы точек спины для модели runs_1
SPINE_INDICES = [1, 2, 3, 4, 5]  # neck, start, middle, hook, end

# Словарь: нога -> пара индексов (верх, низ)
LEG_POINTS_MAP = {
    "left_front": (6, 7),     # left_front_tp, left_front_bt
    "right_front": (8, 9),    # right_front_tp, right_front_bt
    "left_back": (10, 11),    # left_back_tp, left_back_bt
    "right_back": (12, 13)    # right_back_tp, right_back_bt
}

# Короткие названия ног для отображения
LEG_SHORT_NAMES = {
    "left_front": "LF",
    "right_front": "RF",
    "left_back": "LB",
    "right_back": "RB"
}

# Индексы для расчета длины тела
NECK_INDEX = 1
END_INDEX = 5

# Настройки трекинга движения
MOVEMENT_CONFIG = {
    "max_history_length": 5,
    "movement_threshold": 0.6,
    "timeout_seconds": 2,
    "leg_history_length": 3,
    "keypoint_confidence_threshold": 0.3,
}

# Пороги для определения движения
THRESHOLD_CONFIG = {
    "threshold_relative": 0.03,  # порог для движения ног (3% от длины тела)
    "threshold_stable": 0.01,    # порог для движения спины
}

# Настройки визуализации
VISUALIZATION_CONFIG = {
    "moving_color": (0, 255, 0),      # Зеленый - движется
    "stopped_color": (0, 0, 255),     # Красный - стоит
    "line_thickness": 3,
    "keypoint_radius": 5,
    "keypoint_border_thickness": 1,
    "text_font": 0,  # cv2.FONT_HERSHEY_SIMPLEX
    "text_scale": 0.6,
    "text_thickness": 2,
    "label_text_scale": 0.7,
    "stats_text_scale": 0.7,
}

# Настройки YOLO
YOLO_CONFIG = {
    "conf_threshold": 0.8,
    "tracker": "bytetrack.yaml",
}

# Настройки видео
VIDEO_CONFIG = {
    "fourcc": "mp4v",
    "progress_update_interval": 30,  # кадров
}

# Настройки активации обработки коровы
ACTIVATION_CONFIG = {
    "enable_leg_check": True,          # Включить проверку появления ног
    "leg_confidence_threshold": 0.3,   # Минимальная уверенность для точек ног
    "require_all_legs": False,         # False = хотя бы одна нога, True = все 4 ноги
}

# Пути к файлам (можно изменить здесь вместо редактирования кода)
PATHS_CONFIG = {
    # Путь к обученной модели YOLO (.pt файл)
    "model_path": "models/y11n-002.pt",
    
    # Путь к входному видео для обработки
    "video_path": "examples/test2.mp4",
    
    # Путь для сохранения результата (None = автоматически создастся рядом с видео)
    "output_path": None,
}
