"""
Скрипт для генерации демонстрационного датасета RecGym
Создаёт CSV файл с синтетическими данными для тестирования приложения
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def generate_recgym_demo_data(num_samples=10000, output_file='data/recgym_demo.csv'):
    """
    Генерация демонстрационных данных RecGym
    
    Args:
        num_samples (int): Количество записей
        output_file (str): Путь для сохранения CSV
    """
    print(f"🔄 Генерация {num_samples} записей демо-датасета RecGym...")
    
    exercises = ['Adductor', 'ArmCurl', 'BenchPress', 'LegCurl', 'LegPress', 
                 'Ride', 'RopeJumping', 'Run', 'Squat', 'StairUp', 'Walk', 'Null']
    
    positions = ['Wrist', 'Pocket', 'Calf']
    subjects = list(range(1, 11))
    sessions = list(range(1, 6))
    
    data = []
    
    for i in tqdm(range(num_samples), desc="Генерация данных"):
        exercise = np.random.choice(exercises)
        subject = np.random.choice(subjects)
        position = np.random.choice(positions)
        session = np.random.choice(sessions)
        
        # Генерация реалистичных паттернов для каждого упражнения
        if exercise == 'Squat':
            # Приседания: большие изменения по Y, средние по X и Z
            accel = np.random.normal([0, 9.8, 0], [2, 1, 2])
            gyro = np.random.normal([1, 0.5, 0], [0.5, 0.3, 0.2])
            capacitance = np.random.normal(800, 50)
            
        elif exercise == 'Run':
            # Бег: высокие значения ускорения, особенно по Y
            accel = np.random.normal([0, 12, 0], [3, 2, 3])
            gyro = np.random.normal([2, 1, 1], [1, 0.8, 0.8])
            capacitance = np.random.normal(750, 60)
            
        elif exercise == 'BenchPress':
            # Жим лёжа: изменения в горизонтальной плоскости
            accel = np.random.normal([0, 0, 9.8], [1.5, 1.5, 1])
            gyro = np.random.normal([0, 1, 0.5], [0.3, 0.5, 0.3])
            capacitance = np.random.normal(850, 40)
            
        elif exercise == 'Walk':
            # Ходьба: умеренные изменения
            accel = np.random.normal([0, 10, 0], [1, 0.5, 1])
            gyro = np.random.normal([0.3, 0.3, 0.5], [0.2, 0.2, 0.3])
            capacitance = np.random.normal(780, 45)
            
        elif exercise == 'Null':
            # Отдых: минимальные изменения
            accel = np.random.normal([0, 9.8, 0], [0.3, 0.2, 0.3])
            gyro = np.random.normal([0, 0, 0], [0.05, 0.05, 0.05])
            capacitance = np.random.normal(820, 30)
            
        elif exercise == 'RopeJumping':
            # Прыжки на скакалке: высокая частота, большие изменения
            accel = np.random.normal([0, 13, 0], [3.5, 2.5, 3.5])
            gyro = np.random.normal([2.5, 1.5, 1.5], [1.2, 1, 1])
            capacitance = np.random.normal(740, 70)
            
        elif exercise == 'StairUp':
            # Подъём по лестнице: ритмичные изменения по Y
            accel = np.random.normal([0, 11, 0], [2, 1.5, 2])
            gyro = np.random.normal([1.5, 1, 0.8], [0.7, 0.6, 0.5])
            capacitance = np.random.normal(790, 55)
            
        elif exercise == 'Ride':
            # Велосипед: циклические движения ног
            accel = np.random.normal([0, 9.5, 0], [1.5, 1, 1.5])
            gyro = np.random.normal([1, 0.8, 1.2], [0.5, 0.4, 0.6])
            capacitance = np.random.normal(810, 45)
            
        elif exercise == 'ArmCurl':
            # Сгибание рук: движения в вертикальной плоскости
            accel = np.random.normal([0, 8, 2], [1.5, 1, 1.5])
            gyro = np.random.normal([0.5, 1.5, 0.5], [0.3, 0.7, 0.3])
            capacitance = np.random.normal(830, 40)
            
        elif exercise == 'LegPress':
            # Жим ногами: толчковые движения
            accel = np.random.normal([0, 10.5, 0], [2, 1.5, 2])
            gyro = np.random.normal([1, 0.5, 0.5], [0.5, 0.3, 0.3])
            capacitance = np.random.normal(860, 45)
            
        elif exercise == 'LegCurl':
            # Сгибание ног: изолированные движения
            accel = np.random.normal([0, 9, 0], [1.5, 1, 1.5])
            gyro = np.random.normal([0.8, 0.5, 0.8], [0.4, 0.3, 0.4])
            capacitance = np.random.normal(840, 40)
            
        else:  # Adductor
            # Приведение: боковые движения
            accel = np.random.normal([1.5, 9.8, 0], [1.5, 1, 1.5])
            gyro = np.random.normal([0.5, 0.3, 1], [0.3, 0.2, 0.5])
            capacitance = np.random.normal(820, 45)
        
        # Добавление влияния позиции датчика
        if position == 'Wrist':
            # Запястье: больше движений в руках
            gyro = gyro * 1.3
        elif position == 'Pocket':
            # Карман: более сглаженные значения
            accel = accel * 0.8
            gyro = gyro * 0.7
        elif position == 'Calf':
            # Икра: больше движений ног
            accel = accel * 1.1
        
        # Создание записи
        data.append({
            'Subject': subject,
            'Position': position,
            'Session': session,
            'A_x': round(accel[0], 4),
            'A_y': round(accel[1], 4),
            'A_z': round(accel[2], 4),
            'G_x': round(gyro[0], 4),
            'G_y': round(gyro[1], 4),
            'G_z': round(gyro[2], 4),
            'C_1': round(capacitance, 2),
            'Workout': exercise
        })
    
    # Создание DataFrame
    df = pd.DataFrame(data)
    
    # Сохранение в CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Датасет сохранён в: {output_file}")
    print(f"📊 Всего записей: {len(df)}")
    print(f"🏋️ Уникальных упражнений: {df['Workout'].nunique()}")
    print(f"👤 Субъектов: {df['Subject'].nunique()}")
    print(f"📍 Позиций: {df['Position'].nunique()}")
    
    print("\n📈 Распределение упражнений:")
    print(df['Workout'].value_counts().to_string())
    
    return df

if __name__ == "__main__":
    # Создание демо-датасета
    df = generate_recgym_demo_data(
        num_samples=10000,
        output_file='data/recgym_demo.csv'
    )
    
    print("\n✅ Генерация завершена!")
    print("📁 Файл готов к использованию в Streamlit приложении")
