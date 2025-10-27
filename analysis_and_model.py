"""
Страница анализа данных и обучения моделей
Полный пайплайн: загрузка → предобработка → обучение → оценка → предсказания
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import zipfile
import io

warnings.filterwarnings('ignore')

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

st.title("🏋️ Классификация физических упражнений")
st.markdown("---")


# ==================== ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ====================

@st.cache_data
def load_demo_data(num_samples=5000):
    """Генерация демонстрационных данных для тестирования"""
    st.info("📝 Генерируется демо-датасет (5000 записей)...")
    
    exercises = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8']
    data = []
    
    for i in range(num_samples):
        exercise = np.random.choice(exercises)
        
        # Генерация реалистичных паттернов для IMU датчиков
        accel = np.random.normal([0, -9.8, 0], [2, 1.5, 2])
        gyro = np.random.normal([0, 0, 0], [0.5, 0.5, 0.5])
        mag = np.random.normal([0.6, 0.45, -0.08], [0.02, 0.02, 0.02])
        
        data.append({
            'Session': f's{np.random.randint(1, 6)}',
            'Exercise': exercise,
            'User': f'u{np.random.randint(1, 6)}',
            'A_x': accel[0],
            'A_y': accel[1],
            'A_z': accel[2],
            'G_x': gyro[0],
            'G_y': gyro[1],
            'G_z': gyro[2],
            'M_x': mag[0],
            'M_y': mag[1],
            'M_z': mag[2],
            'Workout': exercise
        })
    
    return pd.DataFrame(data)


@st.cache_data
def load_uci_physical_therapy():
    """Загрузка датасета Physical Therapy Exercises"""
    st.warning("⚠️ Этот датасет недоступен для автоматической загрузки через API")
    st.info("""
    📥 Пожалуйста, загрузите датасет вручную:
    
    1. Перейдите на: https://archive.ics.uci.edu/dataset/730/physical+therapy+exercises
    2. Скачайте ZIP архив (45.7 MB)
    3. Выберите опцию "📦 Загрузить ZIP архив" выше
    4. Загрузите скачанный файл
    """)
    return None


@st.cache_data
def load_from_zip(uploaded_zip, sample_rate=1):
    """
    Загрузка данных из ZIP архива с TXT файлами Physical Therapy
    sample_rate: коэффициент прореживания (1=все данные, 2=каждая вторая запись, и т.д.)
    """
    
    try:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            txt_files = [f for f in file_list if f.endswith('.txt') and not f.startswith('__MACOSX')]
            
            st.info(f"📦 Найдено {len(txt_files)} TXT файлов в архиве")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Создаем список для DataFrame по частям
            dataframes = []
            chunk_data = []
            chunk_size = 10000  # Обрабатываем по 10000 записей
            
            files_processed = 0
            total_records = 0
            errors = []
            
            for idx, txt_file in enumerate(txt_files):
                try:
                    # Извлекаем информацию из пути
                    parts = txt_file.replace('\\', '/').split('/')
                    
                    session = None
                    exercise = None  
                    user = None
                    
                    for part in parts:
                        if part.startswith('s') and len(part) <= 3:
                            session = part
                        elif part.startswith('e') and len(part) <= 3:
                            exercise = part
                        elif part.startswith('u') and len(part) <= 3:
                            user = part
                    
                    # Читаем файл
                    with zip_ref.open(txt_file) as f:
                        content = f.read()
                        
                        try:
                            text = content.decode('utf-8')
                        except:
                            try:
                                text = content.decode('latin-1')
                            except:
                                text = content.decode('cp1252', errors='ignore')
                        
                        lines = text.strip().split('\n')
                        
                        for line_idx, line in enumerate(lines):
                            # Прореживание данных
                            if line_idx % sample_rate != 0:
                                continue
                            
                            line = line.strip()
                            
                            if not line or 'time index' in line.lower() or line.startswith('#'):
                                continue
                            
                            values = line.split(';')
                            values = [v.strip() for v in values if v.strip()]
                            
                            if len(values) >= 10:
                                try:
                                    row = {
                                        'Session': session if session else 'unknown',
                                        'Exercise': exercise if exercise else 'unknown',
                                        'User': user if user else 'unknown',
                                        'A_x': float(values[1]),
                                        'A_y': float(values[2]),
                                        'A_z': float(values[3]),
                                        'G_x': float(values[4]),
                                        'G_y': float(values[5]),
                                        'G_z': float(values[6]),
                                        'M_x': float(values[7]),
                                        'M_y': float(values[8]),
                                        'M_z': float(values[9]),
                                        'Workout': exercise if exercise else 'unknown'
                                    }
                                    chunk_data.append(row)
                                    total_records += 1
                                    
                                    # Когда накопили chunk_size записей, создаем DataFrame
                                    if len(chunk_data) >= chunk_size:
                                        dataframes.append(pd.DataFrame(chunk_data))
                                        chunk_data = []
                                        
                                except (ValueError, IndexError):
                                    continue
                        
                        files_processed += 1
                
                except Exception as e:
                    errors.append(f"{txt_file}: {str(e)[:50]}")
                
                # Обновление прогресса
                if idx % 10 == 0 or idx == len(txt_files) - 1:
                    progress = (idx + 1) / len(txt_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Обработано: {idx + 1}/{len(txt_files)} | Записей: {total_records:,}")
            
            # Добавляем последний chunk
            if chunk_data:
                dataframes.append(pd.DataFrame(chunk_data))
            
            progress_bar.empty()
            status_text.empty()
            
            if dataframes:
                st.info("📊 Объединение данных...")
                df = pd.concat(dataframes, ignore_index=True)
                
                st.success(f"✅ Загружено {len(df):,} записей из {files_processed} файлов")
                
                # Статистика
                st.info(f"""
                📊 Статистика загрузки:
                - Файлов обработано: {files_processed}
                - Записей загружено: {len(df):,}
                - Упражнений: {df['Exercise'].nunique()}
                - Пользователей: {df['User'].nunique()}
                - Сессий: {df['Session'].nunique()}
                - Прореживание: каждая {sample_rate}-я запись
                """)
                
                return df
            else:
                st.error("❌ Не удалось извлечь данные")
                return None
            
    except MemoryError:
        st.error("❌ Недостаточно памяти!")
        st.warning("💡 Попробуйте увеличить прореживание данных (sample_rate > 1)")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ==================== ФУНКЦИИ ПРЕДОБРАБОТКИ ====================

def calculate_derived_features(data):
    """Вычисление производных признаков"""
    # Величина ускорения
    data['accel_magnitude'] = np.sqrt(
        data['accel_x']**2 + data['accel_y']**2 + data['accel_z']**2
    )
    
    # Величина угловой скорости
    data['gyro_magnitude'] = np.sqrt(
        data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2
    )
    
    # Производные по времени
    for col in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        data[f'{col}_diff'] = data[col].diff().fillna(0)
    
    return data


def extract_window_features(segment, sensor_columns):
    """Извлечение признаков из временного окна"""
    features = {}
    
    for column in sensor_columns:
        sensor_data = segment[column].values
        
        # Статистические признаки
        features[f'{column}_mean'] = np.mean(sensor_data)
        features[f'{column}_std'] = np.std(sensor_data)
        features[f'{column}_max'] = np.max(sensor_data)
        features[f'{column}_min'] = np.min(sensor_data)
        features[f'{column}_median'] = np.median(sensor_data)
        
        # Признаки распределения
        features[f'{column}_skew'] = stats.skew(sensor_data)
        features[f'{column}_kurtosis'] = stats.kurtosis(sensor_data)
        
        # Энергия
        features[f'{column}_energy'] = np.sum(sensor_data**2) / len(sensor_data)
        
        # Диапазон
        features[f'{column}_range'] = np.max(sensor_data) - np.min(sensor_data)
    
    return features


@st.cache_data
def preprocess_data(data, window_size=50, overlap=0.5):
    """Предобработка данных с сегментацией"""
    
    # Переименование столбцов
    data = data.rename(columns={
        'A_x': 'accel_x', 'A_y': 'accel_y', 'A_z': 'accel_z',
        'G_x': 'gyro_x', 'G_y': 'gyro_y', 'G_z': 'gyro_z',
        'Workout': 'exercise_type'
    })
    
    # Создание timestamp
    data['timestamp'] = data.index
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Вычисление производных признаков
    data = calculate_derived_features(data)
    
    # Сегментация
    step_size = int(window_size * (1 - overlap))
    segments = []
    labels = []
    
    sensor_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                      'accel_magnitude', 'gyro_magnitude']
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_segments = (len(data) - window_size) // step_size + 1
    
    for idx, start_idx in enumerate(range(0, len(data) - window_size + 1, step_size)):
        end_idx = start_idx + window_size
        segment = data.iloc[start_idx:end_idx]
        
        features = extract_window_features(segment, sensor_columns)
        segments.append(features)
        
        # Мажоритарное голосование для метки
        label_counts = segment['exercise_type'].value_counts()
        if not label_counts.empty:
            labels.append(label_counts.index[0])
        
        # Обновление прогресса
        if idx % 10 == 0:
            progress = (idx + 1) / total_segments
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Обработано сегментов: {idx + 1} / {total_segments}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Создание DataFrame
    feature_df = pd.DataFrame(segments)
    feature_df['exercise_type'] = labels
    
    return feature_df
# ==================== ФУНКЦИИ ОБУЧЕНИЯ ====================

def train_model(model_type, X_train, X_test, y_train, y_test):
    """Обучение модели выбранного типа"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    model = models[model_type]
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    
    metrics = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    
    return metrics


def plot_confusion_matrix(cm, classes):
    """Визуализация матрицы ошибок"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title('Матрица ошибок', fontsize=16, fontweight='bold')
    ax.set_xlabel('Предсказанные метки', fontsize=12)
    ax.set_ylabel('Истинные метки', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    """Визуализация важности признаков"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis', ax=ax)
        ax.set_title(f'Топ-{top_n} важных признаков', fontsize=16, fontweight='bold')
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_ylabel('Признак', fontsize=12)
        plt.tight_layout()
        return fig
    return None


def plot_class_accuracy(y_test, y_pred, classes):
    """Визуализация точности по классам"""
    class_accuracy = []
    for class_name in classes:
        class_mask = y_test == class_name
        if np.sum(class_mask) > 0:
            acc = np.mean(y_pred[class_mask] == class_name)
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, class_accuracy, color='skyblue', edgecolor='navy')
    ax.set_title('Точность по классам упражнений', fontsize=16, fontweight='bold')
    ax.set_xlabel('Класс упражнения', fontsize=12)
    ax.set_ylabel('Точность', fontsize=12)
    ax.set_ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    
    # Добавление значений на столбцы
    for bar, acc in zip(bars, class_accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig


# ==================== ИНТЕРФЕЙС ====================

# Секция 1: Загрузка данных
st.header("📁 1. Загрузка данных")

data_source = st.radio(
    "Выберите источник данных:",
    [
        "📦 Загрузить ZIP архив",
        "📄 Загрузить CSV файл",
        "🧪 Использовать демо-данные",
    ],
    horizontal=False,
    index=0
)

# Загрузка данных
data = None

if data_source == "📦 Загрузить ZIP архив":
    uploaded_zip = st.file_uploader("Загрузите ZIP архив с TXT файлами", type=['zip'])
    
    # Добавляем параметр прореживания
    sample_rate = st.slider(
        "Прореживание данных (для экономии памяти)",
        min_value=1, max_value=10, value=2,
        help="1 = все данные, 2 = каждая вторая запись, 10 = каждая десятая"
    )
    
    if uploaded_zip is not None:
        if st.button("🔓 Распаковать и загрузить", type="primary"):
            data = load_from_zip(uploaded_zip, sample_rate)
            if data is not None:
                st.session_state['raw_data'] = data

elif data_source == "📄 Загрузить CSV файл":
    uploaded_file = st.file_uploader("Загрузите CSV файл с датасетом", type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Файл загружен: {uploaded_file.name}")
            st.session_state['raw_data'] = data
        except Exception as e:
            st.error(f"❌ Ошибка загрузки файла: {e}")

elif data_source == "🧪 Использовать демо-данные":
    if st.button("🎲 Сгенерировать демо-данные", type="primary"):
        data = load_demo_data()
        st.session_state['raw_data'] = data

# ВСЕГДА получаем данные из session_state
if 'raw_data' in st.session_state:
    data = st.session_state['raw_data']

# Отображение данных
if data is not None:
    st.subheader("📊 Информация о датасете")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего записей", f"{len(data):,}")
    col2.metric("Признаков", len(data.columns) - 1)
    
    # Определяем колонку с метками
    label_col = None
    for col in ['Workout', 'Exercise', 'exercise_type', 'label', 'class']:
        if col in data.columns:
            label_col = col
            break
    
    if label_col:
        col3.metric("Упражнений", data[label_col].nunique())
    
    # Проверяем наличие информации о субъектах
    subject_col = None
    for col in ['Subject', 'User', 'subject', 'user']:
        if col in data.columns:
            subject_col = col
            break
    
    if subject_col:
        col4.metric("Субъектов", data[subject_col].nunique())
    
    with st.expander("👁️ Просмотр данных"):
        st.dataframe(data.head(100), use_container_width=True)
    
    if label_col:
        with st.expander("📈 Распределение упражнений"):
            workout_counts = data[label_col].value_counts()
            fig, ax = plt.subplots(figsize=(12, 6))
            workout_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='navy')
            ax.set_title('Распределение упражнений в датасете', fontsize=14, fontweight='bold')
            ax.set_xlabel('Упражнение')
            ax.set_ylabel('Количество')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    
    # Секция 2: Предобработка
    st.header("🔄 2. Предобработка данных")
    
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Размер окна (количество записей)", 20, 100, 50, 5)
    with col2:
        overlap = st.slider("Перекрытие окон", 0.0, 0.9, 0.5, 0.1)
    
    if st.button("🚀 Начать предобработку", type="primary"):
        with st.spinner("⏳ Идёт предобработка данных..."):
            processed_data = preprocess_data(data, window_size, overlap)
            st.session_state['processed_data'] = processed_data
            st.success(f"✅ Предобработка завершена! Создано {len(processed_data)} сегментов.")
    
    # Секция 3: Обучение моделей
    if 'processed_data' in st.session_state:
        st.markdown("---")
        st.header("🤖 3. Обучение моделей")
        
        processed_data = st.session_state['processed_data']
        
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Выберите модель:",
                ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost', 'Decision Tree', 'K-Nearest Neighbors']
            )
        
        with col2:
            test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
        
        if st.button("🎯 Обучить модель", type="primary"):
            with st.spinner(f"⏳ Обучение модели {model_type}..."):
                # Подготовка данных
                X = processed_data.drop(columns=['exercise_type'])
                y = processed_data['exercise_type']
                
                # Масштабирование
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Разделение
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Обучение
                metrics = train_model(model_type, X_train, X_test, y_train, y_test)
                
                # Сохранение в session_state
                st.session_state['metrics'] = metrics
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['model_type'] = model_type
                
                st.success(f"✅ Модель {model_type} успешно обучена!")
        
        # Секция 4: Результаты
        if 'metrics' in st.session_state:
            st.markdown("---")
            st.header("📊 4. Результаты обучения")
            
            metrics = st.session_state['metrics']
            model_type = st.session_state['model_type']
            
            # Метрики
            st.subheader(f"📈 Метрики модели: {model_type}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            
            col1, col2 = st.columns(2)
            col1.metric("CV Mean", f"{metrics['cv_mean']:.4f}")
            col2.metric("CV Std", f"{metrics['cv_std']:.4f}")
            
            # Визуализации
            st.subheader("📊 Визуализации")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Матрица ошибок", "Важность признаков", "Точность по классам", "Отчёт"])
            
            with tab1:
                classes = np.unique(st.session_state['y_test'])
                fig_cm = plot_confusion_matrix(metrics['confusion_matrix'], classes)
                st.pyplot(fig_cm)
                plt.close()
            
            with tab2:
                fig_fi = plot_feature_importance(metrics['model'], st.session_state['feature_names'])
                if fig_fi:
                    st.pyplot(fig_fi)
                    plt.close()
                else:
                    st.info("Данная модель не поддерживает отображение важности признаков")
            
            with tab3:
                fig_ca = plot_class_accuracy(st.session_state['y_test'], metrics['y_pred'], classes)
                st.pyplot(fig_ca)
                plt.close()
            
            with tab4:
                st.text("Детальный отчёт по классификации:")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.dataframe(report_df, use_container_width=True)
            
            # Секция 5: Предсказания
            st.markdown("---")
            st.header("🔮 5. Предсказания на новых данных")
            
            st.info("💡 Введите значения датчиков для предсказания типа упражнения")
            
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Акселерометр (м/с²)**")
                    a_x = st.number_input("A_x", value=0.0, format="%.3f")
                    a_y = st.number_input("A_y", value=-9.8, format="%.3f")
                    a_z = st.number_input("A_z", value=0.0, format="%.3f")
                
                with col2:
                    st.markdown("**Гироскоп (рад/с)**")
                    g_x = st.number_input("G_x", value=0.0, format="%.3f")
                    g_y = st.number_input("G_y", value=0.0, format="%.3f")
                    g_z = st.number_input("G_z", value=0.0, format="%.3f")
                
                with col3:
                    st.markdown("**Магнитометр**")
                    m_x = st.number_input("M_x", value=0.6, format="%.3f")
                    m_y = st.number_input("M_y", value=0.45, format="%.3f")
                    m_z = st.number_input("M_z", value=-0.08, format="%.3f")
                
                submit = st.form_submit_button("🎯 Предсказать", type="primary")
                
                if submit:
                    # Создание признаков
                    accel_magnitude = np.sqrt(a_x**2 + a_y**2 + a_z**2)
                    gyro_magnitude = np.sqrt(g_x**2 + g_y**2 + g_z**2)
                    
                    # Создание сегмента данных (дублирование для окна)
                    window_data = pd.DataFrame([{
                        'accel_x': a_x, 'accel_y': a_y, 'accel_z': a_z,
                        'gyro_x': g_x, 'gyro_y': g_y, 'gyro_z': g_z,
                        'accel_magnitude': accel_magnitude,
                        'gyro_magnitude': gyro_magnitude
                    }] * 50)  # Повторяем для создания окна
                    
                    # Извлечение признаков
                    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                                   'accel_magnitude', 'gyro_magnitude']
                    features = extract_window_features(window_data, sensor_cols)
                    
                    # Создание DataFrame с правильным порядком столбцов
                    feature_df = pd.DataFrame([features])
                    feature_df = feature_df[st.session_state['feature_names']]
                    
                    # Масштабирование
                    X_pred = st.session_state['scaler'].transform(feature_df)
                    
                    # Предсказание
                    prediction = metrics['model'].predict(X_pred)[0]
                    
                    if hasattr(metrics['model'], 'predict_proba'):
                        proba = metrics['model'].predict_proba(X_pred)[0]
                        confidence = np.max(proba)
                        
                        st.success(f"### 🎯 Предсказанное упражнение: **{prediction}**")
                        st.info(f"### 💯 Уверенность: **{confidence:.2%}**")
                        
                        st.markdown("#### Вероятности по всем классам:")
                        proba_df = pd.DataFrame({
                            'Упражнение': metrics['model'].classes_,
                            'Вероятность': proba
                        }).sort_values('Вероятность', ascending=False)
                        
                        st.dataframe(proba_df, use_container_width=True)
                        
                        # График вероятностей
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(proba_df['Упражнение'], proba_df['Вероятность'], color='steelblue')
                        ax.set_xlabel('Вероятность')
                        ax.set_title('Вероятности классификации')
                        ax.set_xlim([0, 1])
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.success(f"### 🎯 Предсказанное упражнение: **{prediction}**")

else:
    st.info("👆 Загрузите данные или сгенерируйте демо-датасет для начала работы")