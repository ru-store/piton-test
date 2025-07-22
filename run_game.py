#!/usr/bin/env python3
"""
Скрипт запуска игры "Змейка" с проверкой зависимостей
"""

import sys
import subprocess

def check_pygame():
    """Проверяет наличие pygame и предлагает установку если его нет"""
    try:
        import pygame
        print(f"✓ Pygame найден (версия {pygame.version.ver})")
        return True
    except ImportError:
        print("✗ Pygame не найден")
        print("\nДля установки pygame выполните:")
        print("sudo apt update && sudo apt install -y python3-pygame")
        print("или")
        print("pip install pygame")
        return False

def main():
    print("=== Запуск игры 'Змейка' ===")
    print("Проверка зависимостей...")
    
    if not check_pygame():
        print("\nНе удается запустить игру без pygame")
        print("Запускаю демонстрационную версию вместо основной игры...")
        subprocess.run([sys.executable, "snake_demo.py"])
        return
    
    print("\nВсе зависимости найдены! Запускаю игру...")
    print("Управление:")
    print("  ↑ - движение вверх")
    print("  ↓ - движение вниз") 
    print("  ← - движение влево")
    print("  → - движение вправо")
    print("  ESC - выход из игры")
    print("\nЗакройте это окно и наслаждайтесь игрой!")
    
    try:
        subprocess.run([sys.executable, "snake_game.py"])
    except KeyboardInterrupt:
        print("\n\nИгра прервана пользователем")
    except Exception as e:
        print(f"\n\nОшибка при запуске игры: {e}")
        print("Попробуйте запустить: python3 snake_game.py")

if __name__ == "__main__":
    main()