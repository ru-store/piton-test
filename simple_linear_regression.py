"""
ПРОСТОЕ ОБУЧЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ
===================================

Линейная регрессия - базовый алгоритм машинного обучения для предсказания непрерывных значений.
"""

import numpy as np

class LinearRegression:
    """Простая реализация линейной регрессии"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """Обучение модели"""
        # Инициализация весов
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Градиентный спуск
        m = len(y)
        
        for i in range(self.max_iterations):
            # Предсказание
            y_pred = self.predict(X)
            
            # Вычисление градиентов
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            # Обновление весов
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Вывод ошибки каждые 100 итераций
            if i % 100 == 0:
                mse = np.mean((y_pred - y) ** 2)
                print(f"Итерация {i}: MSE = {mse:.4f}")
    
    def predict(self, X):
        """Предсказание"""
        return np.dot(X, self.weights) + self.bias

def demonstrate_linear_regression():
    """Демонстрация работы линейной регрессии"""
    
    print("=" * 50)
    print("ДЕМОНСТРАЦИЯ ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("=" * 50)
    
    # 1. Простой пример с одной переменной
    print("\n1. ПРОСТОЙ ПРИМЕР: y = 2x + 3 + шум")
    print("-" * 40)
    
    # Генерация данных
    np.random.seed(42)
    X_simple = np.random.uniform(0, 10, 100).reshape(-1, 1)
    y_simple = 2 * X_simple.flatten() + 3 + np.random.normal(0, 1, 100)
    
    # Обучение модели
    model_simple = LinearRegression(learning_rate=0.01, max_iterations=500)
    model_simple.fit(X_simple, y_simple)
    
    print(f"Истинные параметры: вес=2, смещение=3")
    print(f"Найденные параметры: вес={model_simple.weights[0]:.3f}, смещение={model_simple.bias:.3f}")
    
    # Тестирование
    test_X = np.array([[5], [7], [9]])
    predictions = model_simple.predict(test_X)
    print(f"Предсказания для X=[5, 7, 9]: {predictions}")
    
    # 2. Пример с несколькими переменными
    print("\n\n2. МНОГОМЕРНЫЙ ПРИМЕР: цена дома")
    print("-" * 40)
    
    # Генерация данных о домах
    np.random.seed(42)
    n_samples = 100
    
    # Признаки: площадь (м²), количество комнат, возраст дома
    area = np.random.uniform(50, 200, n_samples)
    rooms = np.random.randint(1, 6, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    
    X_multi = np.column_stack([area, rooms, age])
    
    # Цена = 1000*площадь + 5000*комнаты - 100*возраст + шум
    y_multi = (1000 * area + 5000 * rooms - 100 * age + 
               np.random.normal(0, 10000, n_samples))
    
    # Нормализация данных
    X_multi_normalized = (X_multi - np.mean(X_multi, axis=0)) / np.std(X_multi, axis=0)
    y_multi_normalized = (y_multi - np.mean(y_multi)) / np.std(y_multi)
    
    # Обучение модели
    model_multi = LinearRegression(learning_rate=0.1, max_iterations=1000)
    model_multi.fit(X_multi_normalized, y_multi_normalized)
    
    print(f"Веса модели: {model_multi.weights}")
    print(f"Смещение: {model_multi.bias:.3f}")
    
    # Пример предсказания
    test_house = np.array([[120, 3, 10]])  # 120м², 3 комнаты, 10 лет
    test_house_norm = (test_house - np.mean(X_multi, axis=0)) / np.std(X_multi, axis=0)
    pred_norm = model_multi.predict(test_house_norm)
    pred_price = pred_norm * np.std(y_multi) + np.mean(y_multi)
    
    print(f"Предсказанная цена дома (120м², 3 комнаты, 10 лет): {pred_price[0]:,.0f} руб.")

def explain_math():
    """Объяснение математики линейной регрессии"""
    
    print("\n\n" + "=" * 50)
    print("МАТЕМАТИКА ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("=" * 50)
    
    print("""
ОСНОВНАЯ ФОРМУЛА:
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

где:
- y: предсказываемое значение
- w₀: смещение (bias/intercept)
- w₁, w₂, ..., wₙ: веса (коэффициенты)
- x₁, x₂, ..., xₙ: входные признаки

ФУНКЦИЯ ПОТЕРЬ (MSE):
L(w) = (1/2m) * Σ(y_pred - y_true)²

где m - количество образцов

ГРАДИЕНТНЫЙ СПУСК:
Для обновления весов используем градиенты:

∂L/∂w = (1/m) * Xᵀ(y_pred - y_true)
∂L/∂b = (1/m) * Σ(y_pred - y_true)

Обновление:
w = w - α * ∂L/∂w
b = b - α * ∂L/∂b

где α - скорость обучения (learning rate)
""")

def practical_tips():
    """Практические советы"""
    
    print("\n" + "=" * 50)
    print("ПРАКТИЧЕСКИЕ СОВЕТЫ")
    print("=" * 50)
    
    print("""
1. ПОДГОТОВКА ДАННЫХ:
   - Нормализуйте признаки (особенно если они в разных масштабах)
   - Проверьте на выбросы
   - Убедитесь в линейной зависимости

2. ВЫБОР ПАРАМЕТРОВ:
   - Learning rate: обычно 0.01-0.1
   - Слишком большой → модель не сходится
   - Слишком маленький → медленное обучение

3. ОЦЕНКА КАЧЕСТВА:
   - MSE (Mean Squared Error)
   - R² (коэффициент детерминации)
   - MAE (Mean Absolute Error)

4. КОГДА ИСПОЛЬЗОВАТЬ:
   - Простые задачи регрессии
   - Базовая модель для сравнения
   - Когда нужна интерпретируемость

5. ОГРАНИЧЕНИЯ:
   - Только линейные зависимости
   - Чувствительность к выбросам
   - Предполагает нормальное распределение ошибок
""")

if __name__ == "__main__":
    demonstrate_linear_regression()
    explain_math()
    practical_tips()
    
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)
    print("Теперь вы знаете основы линейной регрессии!")
    print("Попробуйте модифицировать код и поэкспериментировать с параметрами.")