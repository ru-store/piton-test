"""
ОБУЧЕНИЕ АЛГОРИТМУ ЛИНЕЙНОЙ РЕГРЕССИИ
=====================================

Линейная регрессия - это один из самых простых и фундаментальных алгоритмов машинного обучения.
Он позволяет предсказывать непрерывные значения на основе входных данных.

ТЕОРИЯ:
-------
Линейная регрессия пытается найти линейную зависимость между входными переменными (X) 
и целевой переменной (y) в виде: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

где:
- w₀ - смещение (bias/intercept)
- w₁, w₂, ..., wₙ - веса (коэффициенты)
- x₁, x₂, ..., xₙ - входные признаки

МАТЕМАТИКА:
-----------
Цель: найти оптимальные веса w, которые минимизируют функцию потерь.
Функция потерь (MSE - Mean Squared Error): L(w) = (1/2m) * Σ(y_pred - y_true)²

Градиентный спуск для обновления весов:
w = w - α * ∇L(w)
где α - скорость обучения (learning rate)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import seaborn as sns

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
np.random.seed(42)

class LinearRegression:
    """
    Реализация линейной регрессии с нуля
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        """
        Инициализация параметров
        
        Args:
            learning_rate: скорость обучения
            max_iterations: максимальное количество итераций
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Обучение модели
        
        Args:
            X: матрица признаков (m, n)
            y: вектор целевых значений (m,)
        """
        # Инициализация параметров
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        # Градиентный спуск
        for i in range(self.max_iterations):
            # Прямой проход (предсказание)
            y_pred = self.predict(X)
            
            # Вычисление функции потерь
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Вычисление градиентов
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            # Обновление параметров
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        Предсказание
        
        Args:
            X: матрица признаков
            
        Returns:
            Предсказанные значения
        """
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true, y_pred):
        """
        Вычисление функции потерь (MSE)
        """
        return np.mean((y_pred - y_true) ** 2)
    
    def score(self, X, y):
        """
        Вычисление коэффициента детерминации R²
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

def demonstrate_simple_linear_regression():
    """
    Демонстрация простой линейной регрессии (одна переменная)
    """
    print("=" * 60)
    print("1. ПРОСТАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ (одна переменная)")
    print("=" * 60)
    
    # Генерация данных
    np.random.seed(42)
    X = np.random.randn(100, 1) * 2
    y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5  # y = 3x + 2 + шум
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"Истинные параметры: вес = 3, смещение = 2")
    print(f"Найденные параметры: вес = {model.weights[0]:.3f}, смещение = {model.bias:.3f}")
    print(f"R² на обучающей выборке: {model.score(X_train, y_train):.3f}")
    print(f"R² на тестовой выборке: {model.score(X_test, y_test):.3f}")
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # График данных и регрессии
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Обучающие данные')
    plt.scatter(X_test, y_test, alpha=0.6, label='Тестовые данные')
    
    # Линия регрессии
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', label='Линия регрессии', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Простая линейная регрессия')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График функции потерь
    plt.subplot(1, 3, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Итерации')
    plt.ylabel('Функция потерь (MSE)')
    plt.title('Сходимость алгоритма')
    plt.grid(True, alpha=0.3)
    
    # График остатков
    plt.subplot(1, 3, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('График остатков')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_multiple_linear_regression():
    """
    Демонстрация множественной линейной регрессии
    """
    print("\n" + "=" * 60)
    print("2. МНОЖЕСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("=" * 60)
    
    # Генерация данных с несколькими признаками
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Найденные веса: {model.weights}")
    print(f"Смещение: {model.bias:.3f}")
    print(f"R² на обучающей выборке: {model.score(X_train, y_train):.3f}")
    print(f"R² на тестовой выборке: {model.score(X_test, y_test):.3f}")
    
    # Визуализация
    plt.figure(figsize=(15, 5))
    
    # Предсказания vs истинные значения
    plt.subplot(1, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, label='Обучающие')
    plt.scatter(y_test, y_test_pred, alpha=0.6, label='Тестовые')
    min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальная линия')
    plt.xlabel('Истинные значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Предсказания vs Истинные значения')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График функции потерь
    plt.subplot(1, 3, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Итерации')
    plt.ylabel('Функция потерь (MSE)')
    plt.title('Сходимость алгоритма')
    plt.grid(True, alpha=0.3)
    
    # Важность признаков
    plt.subplot(1, 3, 3)
    feature_importance = np.abs(model.weights)
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Признаки')
    plt.ylabel('|Вес|')
    plt.title('Важность признаков')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_regularization():
    """
    Демонстрация регуляризации (Ridge регрессия)
    """
    print("\n" + "=" * 60)
    print("3. РЕГУЛЯРИЗАЦИЯ (Ridge регрессия)")
    print("=" * 60)
    
    class RidgeRegression(LinearRegression):
        """
        Ridge регрессия с L2 регуляризацией
        """
        def __init__(self, learning_rate=0.01, max_iterations=1000, alpha=0.1):
            super().__init__(learning_rate, max_iterations)
            self.alpha = alpha  # коэффициент регуляризации
        
        def fit(self, X, y):
            m, n = X.shape
            self.weights = np.zeros(n)
            self.bias = 0
            
            for i in range(self.max_iterations):
                y_pred = self.predict(X)
                
                # Функция потерь с регуляризацией
                cost = self._compute_cost(y, y_pred) + self.alpha * np.sum(self.weights ** 2)
                self.cost_history.append(cost)
                
                # Градиенты с регуляризацией
                dw = (1/m) * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights
                db = (1/m) * np.sum(y_pred - y)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    
    # Создание данных с переобучением
    np.random.seed(42)
    X = np.random.randn(50, 10)  # мало данных, много признаков
    true_weights = np.array([2, -1, 0.5, 0, 0, 1, 0, 0, -0.5, 0])
    y = np.dot(X, true_weights) + np.random.randn(50) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Обычная линейная регрессия
    linear_model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    linear_model.fit(X_train, y_train)
    
    # Ridge регрессия
    ridge_model = RidgeRegression(learning_rate=0.01, max_iterations=1000, alpha=0.1)
    ridge_model.fit(X_train, y_train)
    
    print("Сравнение обычной регрессии и Ridge:")
    print(f"Линейная регрессия - R² train: {linear_model.score(X_train, y_train):.3f}, R² test: {linear_model.score(X_test, y_test):.3f}")
    print(f"Ridge регрессия - R² train: {ridge_model.score(X_train, y_train):.3f}, R² test: {ridge_model.score(X_test, y_test):.3f}")
    
    # Визуализация весов
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    x_pos = np.arange(len(true_weights))
    width = 0.25
    
    plt.bar(x_pos - width, true_weights, width, label='Истинные веса', alpha=0.7)
    plt.bar(x_pos, linear_model.weights, width, label='Линейная регрессия', alpha=0.7)
    plt.bar(x_pos + width, ridge_model.weights, width, label='Ridge регрессия', alpha=0.7)
    
    plt.xlabel('Признаки')
    plt.ylabel('Значения весов')
    plt.title('Сравнение весов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График функций потерь
    plt.subplot(1, 2, 2)
    plt.plot(linear_model.cost_history, label='Линейная регрессия')
    plt.plot(ridge_model.cost_history, label='Ridge регрессия')
    plt.xlabel('Итерации')
    plt.ylabel('Функция потерь')
    plt.title('Сходимость алгоритмов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def practical_example():
    """
    Практический пример: предсказание цен на жилье
    """
    print("\n" + "=" * 60)
    print("4. ПРАКТИЧЕСКИЙ ПРИМЕР: ПРЕДСКАЗАНИЕ ЦЕН НА ЖИЛЬЕ")
    print("=" * 60)
    
    # Генерация синтетических данных о недвижимости
    np.random.seed(42)
    n_samples = 500
    
    # Признаки: площадь, количество комнат, возраст дома, расстояние до центра
    area = np.random.normal(100, 30, n_samples)  # площадь в м²
    rooms = np.random.poisson(3, n_samples) + 1   # количество комнат
    age = np.random.exponential(10, n_samples)    # возраст дома в годах
    distance = np.random.exponential(5, n_samples)  # расстояние до центра в км
    
    # Формула цены (реалистичная зависимость)
    price = (area * 1000 +           # 1000 за м²
             rooms * 15000 +         # 15000 за комнату
             -age * 500 +            # -500 за год возраста
             -distance * 2000 +      # -2000 за км от центра
             50000 +                 # базовая цена
             np.random.normal(0, 10000, n_samples))  # шум
    
    # Подготовка данных
    X = np.column_stack([area, rooms, age, distance])
    feature_names = ['Площадь (м²)', 'Комнаты', 'Возраст (лет)', 'Расстояние (км)']
    
    # Нормализация признаков
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, price, test_size=0.2, random_state=42)
    
    # Обучение модели
    model = LinearRegression(learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("Результаты модели:")
    print(f"R² на обучающей выборке: {model.score(X_train, y_train):.3f}")
    print(f"R² на тестовой выборке: {model.score(X_test, y_test):.3f}")
    print(f"RMSE на тестовой выборке: {np.sqrt(np.mean((y_test - y_test_pred) ** 2)):.0f} руб.")
    
    print("\nВклад признаков в цену:")
    for i, (name, weight) in enumerate(zip(feature_names, model.weights)):
        print(f"{name}: {weight:.0f}")
    
    # Визуализация
    plt.figure(figsize=(15, 10))
    
    # Предсказания vs истинные значения
    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Истинная цена')
    plt.ylabel('Предсказанная цена')
    plt.title('Предсказания vs Истинные цены')
    plt.grid(True, alpha=0.3)
    
    # Важность признаков
    plt.subplot(2, 3, 2)
    plt.bar(range(len(model.weights)), np.abs(model.weights))
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.ylabel('|Вес|')
    plt.title('Важность признаков')
    plt.grid(True, alpha=0.3)
    
    # Распределение остатков
    plt.subplot(2, 3, 3)
    residuals = y_test - y_test_pred
    plt.hist(residuals, bins=20, alpha=0.7)
    plt.xlabel('Остатки')
    plt.ylabel('Частота')
    plt.title('Распределение остатков')
    plt.grid(True, alpha=0.3)
    
    # Зависимость цены от каждого признака
    for i in range(4):
        plt.subplot(2, 3, 4 + i//2 if i < 2 else 5 + i//2)
        if i < 2:
            plt.subplot(2, 3, 4 + i)
        else:
            plt.subplot(2, 3, 5 + (i-2))
        
        # Берем исходные (не нормализованные) данные для наглядности
        original_feature = X[:, i]
        plt.scatter(original_feature, price, alpha=0.3)
        plt.xlabel(feature_names[i])
        plt.ylabel('Цена')
        plt.title(f'Цена vs {feature_names[i]}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Пример предсказания для новой квартиры
    print("\n" + "="*50)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ:")
    print("="*50)
    
    # Новая квартира: 75 м², 2 комнаты, 5 лет, 3 км от центра
    new_apartment = np.array([[75, 2, 5, 3]])
    
    # Нормализуем по статистикам обучающей выборки
    new_apartment_normalized = (new_apartment - X.mean(axis=0)) / X.std(axis=0)
    
    predicted_price = model.predict(new_apartment_normalized)[0]
    
    print(f"Квартира: 75 м², 2 комнаты, 5 лет, 3 км от центра")
    print(f"Предсказанная цена: {predicted_price:.0f} руб.")

def main():
    """
    Главная функция для запуска всех демонстраций
    """
    print("ИЗУЧЕНИЕ АЛГОРИТМА ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("=====================================")
    print("Этот скрипт покажет вам:")
    print("1. Простую линейную регрессию")
    print("2. Множественную линейную регрессию") 
    print("3. Регуляризацию (Ridge)")
    print("4. Практический пример")
    print()
    
    # Запуск всех демонстраций
    demonstrate_simple_linear_regression()
    demonstrate_multiple_linear_regression()
    demonstrate_regularization()
    practical_example()
    
    print("\n" + "="*60)
    print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
    print("="*60)
    print("1. Линейная регрессия ищет линейную зависимость y = Xw + b")
    print("2. Использует градиентный спуск для минимизации MSE")
    print("3. R² показывает качество модели (ближе к 1 = лучше)")
    print("4. Регуляризация помогает избежать переобучения")
    print("5. Важно нормализовать данные для лучшей сходимости")
    print("6. Анализ остатков помогает понять качество модели")

if __name__ == "__main__":
    main()