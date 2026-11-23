#!/usr/bin/env python3
"""
Главный скрипт для запуска обработки видео с использованием YOLO Pose модели
и ByteTracker для отслеживания коров с визуализацией keypoints

Может работать в двух режимах:
1. CLI режим: python main.py --video path/to/video.mp4 --model path/to/model.pt
2. Режим с конфигом: python main.py (используются пути из config/settings.py)
"""

import argparse
import sys
from pathlib import Path

from src import VideoProcessor
from config.settings import PATHS_CONFIG, YOLO_CONFIG


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Обработка видео с детекцией и отслеживанием движения коров"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help=f"Путь к входному видео файлу (по умолчанию: {PATHS_CONFIG['video_path']})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Путь к обученной модели YOLO (.pt файл) (по умолчанию: {PATHS_CONFIG['model_path']})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения выходного видео (по умолчанию: <video>_processed.mp4)"
    )
    
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"Порог уверенности для детекций (по умолчанию: {YOLO_CONFIG['conf_threshold']})"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить отладочный вывод для каждой коровы"
    )
    
    return parser.parse_args()


def main():
    """Основная функция"""
    args = parse_args()
    
    # Определяем пути: из CLI или из конфига
    video_path = args.video if args.video else PATHS_CONFIG['video_path']
    model_path = args.model if args.model else PATHS_CONFIG['model_path']
    output_path = args.output if args.output else PATHS_CONFIG['output_path']
    conf_threshold = args.conf if args.conf is not None else YOLO_CONFIG['conf_threshold']
    
    # Информируем пользователя о режиме работы
    if args.video or args.model:
        print("🔧 Режим работы: CLI (аргументы командной строки)")
    else:
        print("⚙️  Режим работы: Конфигурация (config/settings.py)")
    
    print(f"📹 Видео: {video_path}")
    print(f"🤖 Модель: {model_path}")
    print(f"🎯 Порог уверенности: {conf_threshold}")
    
    # Проверяем существование файлов
    if not Path(video_path).exists():
        print(f"\n❌ Ошибка: видео файл не найден: {video_path}")
        sys.exit(1)
    
    if not Path(model_path).exists():
        print(f"\n❌ Ошибка: файл модели не найден: {model_path}")
        sys.exit(1)
    
    # Создаем процессор видео
    processor = VideoProcessor(
        model_path=model_path,
        conf_threshold=conf_threshold
    )
    
    # Обрабатываем видео
    processor.process_video(
        video_path=video_path,
        output_path=output_path,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
