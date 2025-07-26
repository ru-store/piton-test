"""
ПРАКТИЧЕСКИЕ УПРАЖНЕНИЯ ПО ЛИНЕЙНОЙ РЕГРЕССИИ
============================================

Попробуйте выполнить эти упражнения, чтобы лучше понять алгоритм.
"""

import numpy as np
from simple_linear_regression import LinearRegression

def exercise_1():
    """
    УПРАЖНЕНИЕ 1: Найдите оптимальную скорость обучения
    """
    print("=" * 60)
    print("УПРАЖНЕНИЕ 1: Оптимальная скорость обучения")
    print("=" * 60)
    
    # Генерация данных
    np.random.seed(42)
    X = np.random.uniform(0, 10, 50).reshape(-1, 1)
    y = 3 * X.flatten() + 5 + np.random.normal(0, 2, 50)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    print("Задача: Найти y = 3x + 5")
    print("Тестируем разные скорости обучения:\n")
    
    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr, max_iterations=200)
        print(f"Learning rate: {lr}")
        model.fit(X, y)
        print(f"Результат: вес={model.weights[0]:.3f}, смещение={model.bias:.3f}")
        print(f"Ошибка веса: {abs(model.weights[0] - 3):.3f}")
        print(f"Ошибка смещения: {abs(model.bias - 5):.3f}")
        print("-" * 40)

def exercise_2():
    """
    УПРАЖНЕНИЕ 2: Влияние количества данных на качество
    """
    print("\n" + "=" * 60)
    print("УПРАЖНЕНИЕ 2: Влияние размера данных")
    print("=" * 60)
    
    np.random.seed(42)
    sample_sizes = [10, 50, 100, 500]
    
    print("Истинная функция: y = 1.5x - 2")
    print("Проверяем как размер выборки влияет на точность:\n")
    
    for size in sample_sizes:
        X = np.random.uniform(0, 10, size).reshape(-1, 1)
        y = 1.5 * X.flatten() - 2 + np.random.normal(0, 1, size)
        
        model = LinearRegression(learning_rate=0.01, max_iterations=500)
        model.fit(X, y)
        
        weight_error = abs(model.weights[0] - 1.5)
        bias_error = abs(model.bias - (-2))
        
        print(f"Размер выборки: {size}")
        print(f"Найденные параметры: вес={model.weights[0]:.3f}, смещение={model.bias:.3f}")
        print(f"Ошибки: вес={weight_error:.3f}, смещение={bias_error:.3f}")
        print("-" * 40)

def exercise_3():
    """
    УПРАЖНЕНИЕ 3: Работа с выбросами
    """
    print("\n" + "=" * 60)
    print("УПРАЖНЕНИЕ 3: Влияние выбросов")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Данные без выбросов
    X_clean = np.random.uniform(0, 10, 50).reshape(-1, 1)
    y_clean = 2 * X_clean.flatten() + 1 + np.random.normal(0, 0.5, 50)
    
    # Данные с выбросами
    X_outliers = np.copy(X_clean)
    y_outliers = np.copy(y_clean)
    # Добавляем несколько выбросов
    y_outliers[-5:] += 20  # Сильные выбросы
    
    print("Сравнение модели с выбросами и без:")
    print("Истинная функция: y = 2x + 1\n")
    
    # Модель без выбросов
    model_clean = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model_clean.fit(X_clean, y_clean)
    
    # Модель с выбросами
    model_outliers = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model_outliers.fit(X_outliers, y_outliers)
    
    print("БЕЗ ВЫБРОСОВ:")
    print(f"Параметры: вес={model_clean.weights[0]:.3f}, смещение={model_clean.bias:.3f}")
    
    print("\nС ВЫБРОСАМИ:")
    print(f"Параметры: вес={model_outliers.weights[0]:.3f}, смещение={model_outliers.bias:.3f}")
    
    print(f"\nВлияние выбросов:")
    print(f"Изменение веса: {abs(model_outliers.weights[0] - model_clean.weights[0]):.3f}")
    print(f"Изменение смещения: {abs(model_outliers.bias - model_clean.bias):.3f}")

def exercise_4():
    """
    УПРАЖНЕНИЕ 4: Предсказание зарплаты по опыту
    """
    print("\n" + "=" * 60)
    print("УПРАЖНЕНИЕ 4: Практическая задача - зарплата")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Генерация реалистичных данных
    experience = np.random.uniform(0, 20, 100)  # Опыт работы в годах
    base_salary = 30000  # Базовая зарплата
    salary_per_year = 5000  # Прибавка за год опыта
    
    salary = (base_salary + salary_per_year * experience + 
              np.random.normal(0, 8000, 100))  # Зарплата с шумом
    
    # Подготовка данных
    X = experience.reshape(-1, 1)
    y = salary
    
    # Обучение модели
    model = LinearRegression(learning_rate=0.000001, max_iterations=2000)
    model.fit(X, y)
    
    print("Задача: Предсказать зарплату по опыту работы")
    print(f"Базовая зарплата (теория): {base_salary}")
    print(f"Прибавка за год (теория): {salary_per_year}")
    print()
    print(f"Найденная базовая зарплата: {model.bias:.0f}")
    print(f"Найденная прибавка за год: {model.weights[0]:.0f}")
    print()
    
    # Примеры предсказаний
    test_experience = [1, 5, 10, 15]
    print("Предсказания зарплаты:")
    for exp in test_experience:
        pred_salary = model.predict(np.array([[exp]]))[0]
        print(f"Опыт {exp} лет: {pred_salary:,.0f} руб.")

def bonus_exercise():
    """
    БОНУСНОЕ УПРАЖНЕНИЕ: Сравнение с библиотечной реализацией
    """
    print("\n" + "=" * 60)
    print("БОНУС: Сравнение с scikit-learn")
    print("=" * 60)
    
    try:
        from sklearn.linear_model import LinearRegression as SKLinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        np.random.seed(42)
        X = np.random.uniform(0, 10, 100).reshape(-1, 1)
        y = 3 * X.flatten() + 2 + np.random.normal(0, 1, 100)
        
        # Наша реализация
        our_model = LinearRegression(learning_rate=0.01, max_iterations=1000)
        our_model.fit(X, y)
        our_pred = our_model.predict(X)
        
        # Scikit-learn
        sk_model = SKLinearRegression()
        sk_model.fit(X, y)
        sk_pred = sk_model.predict(X)
        
        print("Сравнение результатов:")
        print(f"Наша модель    - вес: {our_model.weights[0]:.3f}, смещение: {our_model.bias:.3f}")
        print(f"Scikit-learn   - вес: {sk_model.coef_[0]:.3f}, смещение: {sk_model.intercept_:.3f}")
        
        # Метрики качества
        our_mse = mean_squared_error(y, our_pred)
        sk_mse = mean_squared_error(y, sk_pred)
        our_r2 = r2_score(y, our_pred)
        sk_r2 = r2_score(y, sk_pred)
        
        print(f"\nКачество предсказаний:")
        print(f"Наша модель    - MSE: {our_mse:.3f}, R²: {our_r2:.3f}")
        print(f"Scikit-learn   - MSE: {sk_mse:.3f}, R²: {sk_r2:.3f}")
        
    except ImportError:
        print("Scikit-learn не установлен. Пропускаем сравнение.")

def main():
    """Запуск всех упражнений"""
    print("ПРАКТИЧЕСКИЕ УПРАЖНЕНИЯ ПО ЛИНЕЙНОЙ РЕГРЕССИИ")
    print("Выполните упражнения чтобы лучше понять алгоритм!\n")
    
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    bonus_exercise()
    
    print("\n" + "=" * 60)
    print("ВСЕ УПРАЖНЕНИЯ ЗАВЕРШЕНЫ!")
    print("=" * 60)
    print("""
Теперь попробуйте:
1. Изменить параметры и посмотреть как это влияет на результат
2. Создать свои данные и проверить модель
3. Добавить больше признаков в многомерную модель
4. Реализовать другие метрики качества (R², MAE)
""")

if __name__ == "__main__":
    main()