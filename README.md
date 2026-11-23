# 🐄 Cow Video Processor

Автоматическая обработка видео для детекции и отслеживания движения коров с использованием YOLO Pose и ByteTracker.

## 📋 Описание

Этот проект предназначен для анализа видео с коровами и определения их состояния (движение/стоит) с использованием технологий компьютерного зрения. Система использует модель YOLO Pose для детекции ключевых точек (keypoints) на теле коровы и ByteTracker для отслеживания каждой коровы на протяжении всего видео.

### Основные возможности

- ✅ **Детекция коров** с использованием YOLO Pose модели
- ✅ **Отслеживание коров** между кадрами (ByteTracker)
- ✅ **Определение состояния движения** для каждой коровы (MOVE/STOP)
- ✅ **Визуализация ключевых точек** на теле коровы (14 точек)
- ✅ **Цветовая индикация движения ног** (зелёный - движется, красный - стоит)
- ✅ **Отображение статистики** в реальном времени (номер кадра, FPS)
- ✅ **Автоматическая активация** отслеживания при появлении ног в кадре

## 🎯 Архитектура ключевых точек

Модель определяет 14 ключевых точек на теле коровы:

```
Точки тела (спина):
  0: head         - голова
  1: neck         - шея
  2: start        - начало спины
  3: middle       - середина спины
  4: hook         - точка крестца
  5: end          - конец спины

Точки ног:
  6: left_front_tp   - левая передняя нога (верх)
  7: left_front_bt   - левая передняя нога (низ)
  8: right_front_tp  - правая передняя нога (верх)
  9: right_front_bt  - правая передняя нога (низ)
 10: left_back_tp    - левая задняя нога (верх)
 11: left_back_bt    - левая задняя нога (низ)
 12: right_back_tp   - правая задняя нога (верх)
 13: right_back_bt   - правая задняя нога (низ)
```

## 🚀 Быстрый старт

### Требования

- Python 3.8+
- CUDA (опционально, для GPU ускорения)

### Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/cow_video_processor.git
cd cow_video_processor
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Убедитесь, что у вас есть:
   - **Обученная модель YOLO** (`.pt` файл) в папке `models/`
   - **Видео для обработки** в папке `examples/`
  
Тестовые видео в папке examples репозитория сгенерированы с помощью модели **Mimimax H3** 

### Использование

#### Режим 1: С конфигурацией (по умолчанию)

Просто запустите скрипт без аргументов:

```bash
python main.py
```

Пути к видео и модели будут взяты из `config/settings.py` (секция `PATHS_CONFIG`).

#### Режим 2: CLI режим

Укажите пути к видео и модели явно:

```bash
python main.py --video examples/test1.mp4 --model models/y11n-002.pt
```

#### Дополнительные параметры

```bash
python main.py \
  --video examples/test1.mp4 \
  --model models/y11n-002.pt \
  --output custom_output.mp4 \
  --conf 0.8 \
  --verbose
```

**Параметры:**
- `--video` - путь к входному видео
- `--model` - путь к модели YOLO (.pt файл)
- `--output` - путь для сохранения результата (опционально)
- `--conf` - порог уверенности для детекций (по умолчанию: 0.8)
- `--verbose` - включить подробный вывод для каждой коровы

## 📁 Структура проекта

```
cow_video_processor/
│
├── config/                      # Конфигурационные файлы
│   ├── __init__.py
│   └── settings.py              # Основные настройки проекта
│
├── src/                         # Исходный код
│   ├── __init__.py
│   ├── video_processor.py       # Основной класс обработки видео
│   └── movement_tracker.py      # Класс отслеживания движения
│
├── utils/                       # Утилиты
│   ├── __init__.py
│   └── visualization.py         # Функции визуализации
│
├── models/                      # Обученные модели YOLO
│   └── y11n-002.pt
│
├── examples/                    # Примеры видео
│   ├── test1.mp4
│   ├── test2.mp4
│   └── visualized/              # Выходные обработанные видео
│       └── test1_processed.mp4
│
├── main.py                      # Главный скрипт запуска
├── requirements.txt             # Зависимости Python
├── .gitignore                   # Игнорируемые файлы Git
└── README.md                    # Документация
```

## ⚙️ Конфигурация

Все настройки проекта находятся в `config/settings.py`:

### Пути к файлам (`PATHS_CONFIG`)

```python
PATHS_CONFIG = {
    "model_path": "models/y11n-002.pt",
    "video_path": "examples/test1.mp4",
    "output_path": None,  # None = автогенерация
}
```

### Настройки YOLO (`YOLO_CONFIG`)

```python
YOLO_CONFIG = {
    "conf_threshold": 0.8,         # Порог уверенности
    "tracker": "bytetrack.yaml",   # Алгоритм трекинга
}
```

### Настройки движения (`MOVEMENT_CONFIG`)

```python
MOVEMENT_CONFIG = {
    "max_history_length": 5,                  # Длина истории для анализа
    "movement_threshold": 0.6,                # Порог определения движения
    "timeout_seconds": 2,                     # Таймаут неактивности
    "leg_history_length": 3,                  # История движения ног
    "keypoint_confidence_threshold": 0.3,     # Порог уверенности точек
}
```

### Настройки активации (`ACTIVATION_CONFIG`)

```python
ACTIVATION_CONFIG = {
    "enable_leg_check": True,          # Проверять появление ног
    "leg_confidence_threshold": 0.3,   # Минимальная уверенность
    "require_all_legs": False,         # False = хотя бы одна нога
}
```

### Настройки визуализации (`VISUALIZATION_CONFIG`)

```python
VISUALIZATION_CONFIG = {
    "moving_color": (0, 255, 0),       # Зелёный - движется
    "stopped_color": (0, 0, 255),      # Красный - стоит
    "line_thickness": 3,
    "keypoint_radius": 5,
    "text_scale": 0.6,
}
```

## 🎨 Визуализация

На выходном видео отображается:

1. **Bounding Box** вокруг каждой коровы:
   - 🟢 Зелёный - корова движется
   - 🔴 Красный - корова стоит

2. **Метка с информацией:**
   ```
   ID: 1 | MOVE | Conf: 0.95
   ID: 2 | STOP | Conf: 0.88
   ```

3. **Ключевые точки** на теле коровы (14 точек разными цветами)

4. **Линии на ногах** с цветовой индикацией движения каждой ноги:
   - LF (Left Front) - левая передняя
   - RF (Right Front) - правая передняя
   - LB (Left Back) - левая задняя
   - RB (Right Back) - правая задняя

5. **Информация о кадре** (верхний левый угол):
   ```
   Frame: 360/481
   FPS: 34.0
   ```



https://github.com/user-attachments/assets/cf818350-4cf7-4e77-8375-65340a5dc383




https://github.com/user-attachments/assets/7ca2e28e-d176-469b-9a52-477d88983785



## 📊 Выходные данные

- **Формат:** MP4 (или оригинальный формат входного видео)
- **Расположение:** `examples/visualized/`
- **Именование:**
  - `{video_name}_processed.mp4` - первый запуск
  - `{video_name}_processed_1.mp4` - второй запуск
  - И так далее...

## 🤖 Обучение модели

### Датасет

Модель YOLO Pose обучалась на датасете:

🔗 **Датасет:** [Cow Pose Dataset на Roboflow](https://app.roboflow.com/penandpencildraft/cow1-vko5c/2)

Датасет содержит изображения коров с размеченными 14 ключевыми точками на теле животного.

### Мониторинг обучения

Процесс обучения отслеживался и логировался с помощью Comet.ml:

📊 **Эксперименты:** [Cow YOLO Training на Comet.ml](https://www.comet.com/brinkinvision/cow-yolo/view/new/panels)

Вы можете посмотреть:
- Графики Loss функций
- Метрики (mAP, Precision, Recall)
- Визуализацию предсказаний
- Гиперпараметры обучения

## 🛠️ Технологии

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - детекция и pose estimation
- **ByteTracker** - отслеживание объектов между кадрами
- **OpenCV** - обработка видео и визуализация
- **NumPy** - вычисления и обработка данных
- **[Roboflow](https://roboflow.com/)** - подготовка датасета
- **[Comet.ml](https://www.comet.com/)** - мониторинг и логирование обучения

