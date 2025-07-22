#!/usr/bin/env python3
"""
Демонстрационная версия игры змейка для проверки логики
Эта версия работает в терминале без графического интерфейса
"""

import random
import time
import os

class SimpleSnake:
    def __init__(self, width=20, height=10):
        self.width = width
        self.height = height
        self.snake = [(5, 5), (4, 5), (3, 5)]  # голова, тело, хвост
        self.direction = (1, 0)  # движение вправо
        self.food = self.generate_food()
        self.score = 0
        
    def generate_food(self):
        while True:
            food = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if food not in self.snake:
                return food
    
    def move(self):
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        # Проверка столкновения со стенами
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            return False
        
        # Проверка столкновения с собой
        if new_head in self.snake:
            return False
        
        self.snake.insert(0, new_head)
        
        # Проверка поедания еды
        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
        else:
            self.snake.pop()
        
        return True
    
    def display(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * (self.width + 2))
        for y in range(self.height):
            print("|", end="")
            for x in range(self.width):
                if (x, y) == self.snake[0]:
                    print("O", end="")  # голова
                elif (x, y) in self.snake:
                    print("o", end="")  # тело
                elif (x, y) == self.food:
                    print("*", end="")  # еда
                else:
                    print(" ", end="")
            print("|")
        print("=" * (self.width + 2))
        print(f"Счет: {self.score}")
        print("Управление: w-вверх, s-вниз, a-влево, d-вправо, q-выход")

def main():
    print("Демонстрация игры 'Змейка'")
    print("Это упрощенная версия для проверки логики")
    print("Нажмите Enter для начала...")
    input()
    
    game = SimpleSnake()
    
    # Демонстрация автоматического движения
    for i in range(10):
        game.display()
        if not game.move():
            print("Игра окончена!")
            break
        time.sleep(0.5)
        
        # Изменяем направление для демонстрации
        if i == 3:
            game.direction = (0, 1)  # вниз
        elif i == 6:
            game.direction = (-1, 0)  # влево
    
    print(f"\nДемонстрация завершена! Финальный счет: {game.score}")
    print("\nОсновная игра с графикой запускается командой:")
    print("python3 snake_game.py")

if __name__ == "__main__":
    main()