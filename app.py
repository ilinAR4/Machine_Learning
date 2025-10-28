"""
Fitness Classifier - Главный файл приложения
Система классификации фитнес-упражнений на основе данных IMU-датчиков
"""

import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Fitness Classifier",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Определение страниц
pages = [
    st.Page("analysis_and_model.py", title="🏋️ Анализ и модель"),
    st.Page("presentation.py", title="📊 Презентация"),
]

# Создание навигации
pg = st.navigation(pages, position="sidebar")

# Добавление информации в сайдбар
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 О проекте")
    st.markdown("""
    **Fitness Classifier** - система для классификации фитнес-упражнений 
    по данным IMU-датчиков (акселерометр + гироскоп + магнитометр).
    
    **Датасет:** UCI Physical Therapy Exercises  
    **Упражнений:** 8  
    **Датчики:** Акселерометр, Гироскоп, Магнитометр  
    """)
    
    st.markdown("---")
    st.markdown("### Возможности")
    st.markdown("""
    - 📁 Загрузка датасета (ZIP/CSV)
    - 🔄 Предобработка данных
    - 🤖 Обучение моделей ML
    - 📊 Визуализация результатов
    - 🔮 Предсказания в реальном времени
    - 📈 Сравнение моделей
    """)

# Запуск выбранной страницы
pg.run()