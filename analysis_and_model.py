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
    
    # Словарь для перевода названий признаков
    feature_translation = {
        'mean': 'среднее',
        'std': 'станд_откл',
        'max': 'максимум',
        'min': 'минимум',
        'median': 'медиана',
        'skew': 'асимметрия',
        'kurtosis': 'эксцесс',
        'energy': 'энергия',
        'range': 'диапазон'
    }
    
    sensor_translation = {
        'accel_x': 'акс_x',
        'accel_y': 'акс_y',
        'accel_z': 'акс_z',
        'gyro_x': 'гир_x',
        'gyro_y': 'гир_y',
        'gyro_z': 'гир_z',
        'accel_magnitude': 'акс_магнитуда',
        'gyro_magnitude': 'гир_магнитуда'
    }
    
    for column in sensor_columns:
        sensor_data = segment[column].values
        
        # Перевод названия датчика
        sensor_name_ru = sensor_translation.get(column, column)
        
        # Проверка на пустые данные
        if len(sensor_data) == 0:
            # Заполняем нулями если нет данных
            for feat_name, feat_name_ru in feature_translation.items():
                features[f'{sensor_name_ru}_{feat_name_ru}'] = 0.0
            continue
        
        # Статистические признаки (заменяем NaN на 0)
        features[f'{sensor_name_ru}_среднее'] = np.nan_to_num(np.mean(sensor_data), nan=0.0)
        features[f'{sensor_name_ru}_станд_откл'] = np.nan_to_num(np.std(sensor_data), nan=0.0)
        features[f'{sensor_name_ru}_максимум'] = np.nan_to_num(np.max(sensor_data), nan=0.0)
        features[f'{sensor_name_ru}_минимум'] = np.nan_to_num(np.min(sensor_data), nan=0.0)
        features[f'{sensor_name_ru}_медиана'] = np.nan_to_num(np.median(sensor_data), nan=0.0)
        
        # Признаки распределения (обработка NaN)
        skew_val = stats.skew(sensor_data)
        features[f'{sensor_name_ru}_асимметрия'] = np.nan_to_num(skew_val, nan=0.0)
        
        kurt_val = stats.kurtosis(sensor_data)
        features[f'{sensor_name_ru}_эксцесс'] = np.nan_to_num(kurt_val, nan=0.0)
        
        # Энергия
        energy_val = np.sum(sensor_data**2) / len(sensor_data) if len(sensor_data) > 0 else 0.0
        features[f'{sensor_name_ru}_энергия'] = np.nan_to_num(energy_val, nan=0.0)
        
        # Диапазон
        range_val = np.max(sensor_data) - np.min(sensor_data)
        features[f'{sensor_name_ru}_диапазон'] = np.nan_to_num(range_val, nan=0.0)
    
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
    
    # Кодируем метки для XGBoost (требует числовые метки)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    model = models[model_type]
    
    # Обучение (XGBoost требует числовые метки)
    if model_type == 'XGBoost':
        model.fit(X_train, y_train_encoded)
    else:
        model.fit(X_train, y_train)
    
    # Предсказания (декодируем обратно для XGBoost)
    if model_type == 'XGBoost':
        y_pred_encoded = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Кросс-валидация (используем правильные метки)
    if model_type == 'XGBoost':
        cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5, n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    
    metrics = {
        'model': model,
        'label_encoder': label_encoder if model_type == 'XGBoost' else None,
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


def create_models_comparison_table(all_models):
    """Создание сравнительной таблицы всех моделей"""
    comparison_data = []
    
    for model_name, metrics in all_models.items():
        # Вычисляем среднюю уверенность (среднюю максимальную вероятность по всем примерам)
        if metrics['y_pred_proba'] is not None:
            avg_confidence = np.mean(np.max(metrics['y_pred_proba'], axis=1))
        else:
            avg_confidence = None
        
        comparison_data.append({
            'Модель': model_name,
            'Точность (Accuracy)': f"{metrics['accuracy']:.4f}",
            'Уверенность (Confidence)': f"{avg_confidence:.4f}" if avg_confidence else "N/A",
            'Точность положит. (Precision)': f"{metrics['precision']:.4f}",
            'Полнота (Recall)': f"{metrics['recall']:.4f}",
            'F1-мера': f"{metrics['f1_score']:.4f}",
            'CV среднее': f"{metrics['cv_mean']:.4f}",
            'CV откл.': f"{metrics['cv_std']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Сортируем по Точности
    df = df.sort_values('Точность (Accuracy)', ascending=False).reset_index(drop=True)
    
    return df


def plot_models_comparison(all_models):
    """Визуализация сравнения моделей"""
    model_names = list(all_models.keys())
    accuracies = [all_models[name]['accuracy'] for name in model_names]
    f1_scores = [all_models[name]['f1_score'] for name in model_names]
    
    # Сортируем по accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='coral', alpha=0.8)
    
    ax.set_xlabel('Модели', fontsize=12, fontweight='bold')
    ax.set_ylabel('Метрика', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение моделей по метрикам', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Добавление значений на столбцы
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
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
        
        # Выбор режима обучения
        training_mode = st.radio(
            "Режим обучения:",
            ["🎯 Одна модель", "📊 Сравнение всех моделей"],
            horizontal=True
        )
        
        test_size = st.slider("Размер тестовой выборки", 0.1, 0.4, 0.2, 0.05)
        
        if training_mode == "🎯 Одна модель":
            # Режим одной модели
            model_type = st.selectbox(
                "Выберите модель:",
                ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost', 'Decision Tree', 'K-Nearest Neighbors']
            )
            
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
                    st.session_state['all_models'] = {model_type: metrics}
                    
                    st.success(f"✅ Модель {model_type} успешно обучена!")
        
        else:
            # Режим сравнения всех моделей
            st.info("💡 Будут обучены все доступные модели для сравнения")
            
            model_types = ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost', 
                          'Decision Tree', 'K-Nearest Neighbors']
            
            if st.button("🚀 Обучить все модели", type="primary"):
                # Подготовка данных один раз
                X = processed_data.drop(columns=['exercise_type'])
                y = processed_data['exercise_type']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Обучение всех моделей
                all_models = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, model_type in enumerate(model_types):
                    status_text.text(f"⏳ Обучение {model_type}... ({idx+1}/{len(model_types)})")
                    
                    try:
                        metrics = train_model(model_type, X_train, X_test, y_train, y_test)
                        all_models[model_type] = metrics
                    except Exception as e:
                        st.warning(f"⚠️ Ошибка при обучении {model_type}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(model_types))
                
                # Сохранение результатов
                st.session_state['all_models'] = all_models
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['scaler'] = scaler
                st.session_state['feature_names'] = X.columns.tolist()
                
                # Сохраняем лучшую модель как основную
                best_model_name = max(all_models.keys(), key=lambda k: all_models[k]['accuracy'])
                st.session_state['metrics'] = all_models[best_model_name]
                st.session_state['model_type'] = best_model_name
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"✅ Все модели обучены! Лучшая модель: {best_model_name}")
        
        # Секция 3.5: Сравнительная таблица моделей (если обучены несколько)
        if 'all_models' in st.session_state and len(st.session_state['all_models']) > 1:
            st.markdown("---")
            st.header("📊 Сравнение моделей")
            
            all_models = st.session_state['all_models']
            
            # Таблица сравнения
            st.subheader("📋 Сравнительная таблица")
            comparison_df = create_models_comparison_table(all_models)
            
            # Подсветка лучших значений
            styled_df = comparison_df.style.highlight_max(
                subset=['Точность (Accuracy)', 'Уверенность (Confidence)', 'Точность положит. (Precision)', 
                        'Полнота (Recall)', 'F1-мера', 'CV среднее'],
                color='lightgreen'
            ).highlight_min(
                subset=['CV откл.'],
                color='lightgreen'
            )
            
            st.dataframe(styled_df, use_container_width=True, height=300)
            
            # График сравнения
            st.subheader("📈 Визуальное сравнение")
            fig_comparison = plot_models_comparison(all_models)
            st.pyplot(fig_comparison)
            plt.close()
            
            # Вывод лучшей модели
            best_model = max(all_models.keys(), key=lambda k: all_models[k]['accuracy'])
            st.success(f"🏆 **Лучшая модель:** {best_model} (Accuracy: {all_models[best_model]['accuracy']:.4f})")
            
            # 📝 Текстовые выводы по заданному шаблону
            st.markdown("---")
            st.subheader("📝 Выводы по результатам сравнения")
            
            # Получаем параметры
            test_size = st.session_state.get('test_size_used', 0.2)
            best_accuracy = all_models[best_model]['accuracy']
            best_precision = all_models[best_model]['precision']
            best_recall = all_models[best_model]['recall']
            best_f1 = all_models[best_model]['f1_score']
            
            # Находим модель с максимальной уверенностью (средняя вероятность правильных предсказаний)
            # Используем CV Mean как показатель стабильности
            best_confidence_model = max(all_models.keys(), key=lambda k: all_models[k]['cv_mean'])
            best_confidence = all_models[best_confidence_model]['cv_mean']
            
            # Генерируем текстовые выводы
            conclusion_text = f"""
При заданном размере тестовой выборки **{test_size:.1%}** ({int(test_size*100)}% от всех данных) проведено сравнение 
{len(all_models)} моделей машинного обучения для классификации физических упражнений.

**Максимальную точность предсказания** продемонстрировала модель **{best_model}** со следующими показателями:
- **Точность (Accuracy)**: {best_accuracy:.4f} ({best_accuracy*100:.2f}%) — доля правильно классифицированных упражнений от общего числа примеров
- **Точность положительных (Precision)**: {best_precision:.4f} ({best_precision*100:.2f}%) — доля истинно положительных предсказаний среди всех положительных
- **Полнота (Recall)**: {best_recall:.4f} ({best_recall*100:.2f}%) — доля найденных положительных примеров от всех положительных в выборке
- **F1-мера (F1-Score)**: {best_f1:.4f} ({best_f1*100:.2f}%) — гармоническое среднее между Precision и Recall

**Максимальную стабильность и уверенность** (среднюю точность при кросс-валидации) показала модель **{best_confidence_model}** 
с показателем **{best_confidence:.4f}** ({best_confidence*100:.2f}%), что говорит о надёжности предсказаний на различных подвыборках данных.

**Уверенность (Confidence)** — это средняя вероятность, с которой модель относит объекты к предсказанным классам (от 0 до 1 или 0-100%). 
Высокая уверенность (>70%) указывает на то, что модель чётко различает признаки упражнений и уверенно делает предсказания.

**Точность (Accuracy)** — это общая метрика качества модели, показывающая процент правильных предсказаний 
на всей тестовой выборке. Рассчитывается как отношение правильно классифицированных примеров к их общему числу.

**Заключение:** {'Модель ' + best_model + ' рекомендуется для использования в системе классификации упражнений.' if best_model == best_confidence_model else f'Модели {best_model} (по точности) и {best_confidence_model} (по стабильности) показали лучшие результаты и рекомендуются для использования.'}
            """
            
            st.info(conclusion_text)
            
            # Дополнительная таблица с расшифровкой метрик
            with st.expander("📖 Расшифровка метрик"):
                st.markdown("""
                **Точность (Accuracy)** — доля правильных предсказаний от всех предсказаний: `(ИП + ИО) / Всего`
                - Формула: `(TP + TN) / (TP + TN + FP + FN)`
                - Показывает общую эффективность модели
                
                **Уверенность (Confidence)** — средняя максимальная вероятность предсказаний
                - Вычисляется как среднее от максимальных вероятностей по всем примерам
                - Показывает, насколько модель уверена в своих предсказаниях
                - Высокая уверенность (>0.7) — модель чётко различает классы
                
                **Точность положительных (Precision)** — доля истинно положительных среди всех положительных предсказаний: `ИП / (ИП + ЛП)`
                - Формула: `TP / (TP + FP)`
                - Отвечает на вопрос: "Из того, что модель назвала положительным, сколько действительно положительные?"
                
                **Полнота (Recall)** — доля найденных положительных от всех реальных положительных: `ИП / (ИП + ЛО)`
                - Формула: `TP / (TP + FN)`
                - Отвечает на вопрос: "Из всех положительных примеров, сколько модель нашла?"
                
                **F1-мера (F1-Score)** — гармоническое среднее Precision и Recall
                - Формула: `2 * (Precision * Recall) / (Precision + Recall)`
                - Баланс между точностью положительных и полнотой
                
                **CV среднее (CV Mean)** — средняя точность при кросс-валидации
                - Точность модели на K различных подвыборках данных
                - Показывает стабильность модели
                
                **CV откл. (CV Std)** — стандартное отклонение кросс-валидации
                - Разброс точности между подвыборками
                - Чем меньше значение, тем стабильнее модель
                
                ---
                **Обозначения:**
                - **ИП (TP)** — Истинно Положительные (True Positive) — правильно предсказанные положительные
                - **ИО (TN)** — Истинно Отрицательные (True Negative) — правильно предсказанные отрицательные
                - **ЛП (FP)** — Ложно Положительные (False Positive) — ошибочно предсказанные как положительные
                - **ЛО (FN)** — Ложно Отрицательные (False Negative) — ошибочно предсказанные как отрицательные
                """)
            
            # Сохраняем test_size для использования в выводах
            if 'X_test' in st.session_state and 'processed_data' in st.session_state:
                total_samples = len(st.session_state['processed_data'])
                test_samples = len(st.session_state['X_test'])
                actual_test_size = test_samples / total_samples if total_samples > 0 else test_size
                st.session_state['test_size_used'] = actual_test_size
            
            # Детальные визуализации для каждой модели
            st.markdown("---")
            st.subheader("🔍 Детальные результаты по моделям")
            st.info("👇 Выберите модель для просмотра детальной визуализации")
            
            # Выбор модели для детального просмотра
            selected_model_name = st.selectbox(
                "Выберите модель для детального анализа:",
                list(all_models.keys()),
                index=list(all_models.keys()).index(best_model)
            )
            
            if selected_model_name:
                selected_metrics = all_models[selected_model_name]
                
                st.markdown(f"### 📊 Детальный анализ: **{selected_model_name}**")
                
                # Метрики выбранной модели
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")
                col2.metric("Precision", f"{selected_metrics['precision']:.4f}")
                col3.metric("Recall", f"{selected_metrics['recall']:.4f}")
                col4.metric("F1-Score", f"{selected_metrics['f1_score']:.4f}")
                
                # Визуализации в табах
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Матрица ошибок", 
                    "🎯 Важность признаков", 
                    "📈 Точность по классам", 
                    "📋 Детальный отчёт"
                ])
                
                classes = np.unique(st.session_state['y_test'])
                
                with tab1:
                    st.markdown(f"**Матрица ошибок для {selected_model_name}**")
                    fig_cm = plot_confusion_matrix(selected_metrics['confusion_matrix'], classes)
                    st.pyplot(fig_cm)
                    plt.close()
                    
                    # Дополнительная информация
                    with st.expander("ℹ️ Интерпретация матрицы ошибок"):
                        st.markdown("""
                        - **Диагональ** — правильные предсказания
                        - **Вне диагонали** — ошибки классификации
                        - Чем темнее цвет на диагонали, тем лучше
                        """)
                
                with tab2:
                    st.markdown(f"**Важность признаков для {selected_model_name}**")
                    fig_fi = plot_feature_importance(selected_metrics['model'], st.session_state['feature_names'])
                    if fig_fi:
                        st.pyplot(fig_fi)
                        plt.close()
                    else:
                        st.info(f"⚠️ Модель {selected_model_name} не поддерживает отображение важности признаков")
                
                with tab3:
                    st.markdown(f"**Точность классификации по упражнениям для {selected_model_name}**")
                    fig_ca = plot_class_accuracy(
                        st.session_state['y_test'], 
                        selected_metrics['y_pred'], 
                        classes
                    )
                    st.pyplot(fig_ca)
                    plt.close()
                    
                    # Дополнительная статистика
                    with st.expander("📊 Статистика по классам"):
                        class_stats = []
                        for class_name in classes:
                            mask = st.session_state['y_test'] == class_name
                            total = np.sum(mask)
                            correct = np.sum(selected_metrics['y_pred'][mask] == class_name)
                            accuracy = correct / total if total > 0 else 0
                            
                            class_stats.append({
                                'Класс': class_name,
                                'Всего примеров': total,
                                'Правильно': correct,
                                'Неправильно': total - correct,
                                'Точность': f"{accuracy:.2%}"
                            })
                        
                        stats_df = pd.DataFrame(class_stats)
                        st.dataframe(stats_df, use_container_width=True)
                
                with tab4:
                    st.markdown(f"**Полный отчёт классификации для {selected_model_name}**")
                    report_df = pd.DataFrame(selected_metrics['classification_report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Средние метрики
                    st.markdown("**📊 Средние значения метрик:**")
                    col1, col2, col3 = st.columns(3)
                    
                    # Извлекаем weighted avg из отчёта
                    if 'weighted avg' in selected_metrics['classification_report']:
                        weighted_avg = selected_metrics['classification_report']['weighted avg']
                        col1.metric("Weighted Precision", f"{weighted_avg['precision']:.4f}")
                        col2.metric("Weighted Recall", f"{weighted_avg['recall']:.4f}")
                        col3.metric("Weighted F1-Score", f"{weighted_avg['f1-score']:.4f}")


        
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
            
            st.info("💡 Введите значения датчиков вручную для предсказания типа упражнения")
            
            # Выбор модели для предсказания (если обучено несколько)
            if 'all_models' in st.session_state and len(st.session_state['all_models']) > 1:
                prediction_model_name = st.selectbox(
                    "Выберите модель для предсказания:",
                    list(st.session_state['all_models'].keys()),
                    key='prediction_model_select'
                )
                prediction_model = st.session_state['all_models'][prediction_model_name]['model']
                st.caption(f"Используется модель: **{prediction_model_name}**")
            else:
                prediction_model = metrics['model']
                prediction_model_name = st.session_state.get('model_type', 'Текущая модель')
            
            with st.form("prediction_form"):
                st.markdown("### 📝 Ввод значений датчиков")
                st.caption("Значения по умолчанию соответствуют состоянию покоя")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Акселерометр (м/с²)**")
                    a_x = st.number_input("A_x", value=0.0, format="%.3f", help="Ускорение по оси X")
                    a_y = st.number_input("A_y", value=-9.8, format="%.3f", help="Ускорение по оси Y (гравитация)")
                    a_z = st.number_input("A_z", value=0.0, format="%.3f", help="Ускорение по оси Z")
                
                with col2:
                    st.markdown("**Гироскоп (рад/с)**")
                    g_x = st.number_input("G_x", value=0.0, format="%.3f", help="Угловая скорость по оси X")
                    g_y = st.number_input("G_y", value=0.0, format="%.3f", help="Угловая скорость по оси Y")
                    g_z = st.number_input("G_z", value=0.0, format="%.3f", help="Угловая скорость по оси Z")
                
                with col3:
                    st.markdown("**Магнитометр (единицы)**")
                    m_x = st.number_input("M_x", value=0.6, format="%.3f", help="Магнитное поле по оси X")
                    m_y = st.number_input("M_y", value=0.45, format="%.3f", help="Магнитное поле по оси Y")
                    m_z = st.number_input("M_z", value=-0.08, format="%.3f", help="Магнитное поле по оси Z")
                
                # Примеры значений
                with st.expander("📋 Примеры значений для разных упражнений"):
                    st.markdown("""
                    **Пример 1: Приседание**
                    - A_x: 0.5, A_y: -8.0, A_z: 2.0
                    - G_x: 0.1, G_y: 0.2, G_z: -0.1
                    - M_x: 0.58, M_y: 0.43, M_z: -0.09
                    
                    **Пример 2: Прыжок**
                    - A_x: 1.5, A_y: -5.0, A_z: 3.5
                    - G_x: 0.8, G_y: 1.2, G_z: 0.5
                    - M_x: 0.62, M_y: 0.47, M_z: -0.07
                    
                    **Пример 3: Махи руками**
                    - A_x: 2.0, A_y: -9.0, A_z: 1.0
                    - G_x: 1.5, G_y: 0.3, G_z: 0.8
                    - M_x: 0.55, M_y: 0.40, M_z: -0.10
                    """)
                
                submit = st.form_submit_button("🎯 Предсказать упражнение", type="primary", use_container_width=True)
                
                if submit:
                    # Отображение введённых значений
                    st.markdown("---")
                    st.markdown("### 📊 Введённые значения датчиков")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Акселерометр X", f"{a_x:.3f} м/с²")
                        st.metric("Акселерометр Y", f"{a_y:.3f} м/с²")
                        st.metric("Акселерометр Z", f"{a_z:.3f} м/с²")
                    with col2:
                        st.metric("Гироскоп X", f"{g_x:.3f} рад/с")
                        st.metric("Гироскоп Y", f"{g_y:.3f} рад/с")
                        st.metric("Гироскоп Z", f"{g_z:.3f} рад/с")
                    with col3:
                        st.metric("Магнитометр X", f"{m_x:.3f}")
                        st.metric("Магнитометр Y", f"{m_y:.3f}")
                        st.metric("Магнитометр Z", f"{m_z:.3f}")
                    
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
                    
                    # Предсказание (с декодированием для XGBoost)
                    if 'all_models' in st.session_state and prediction_model_name in st.session_state['all_models']:
                        model_metrics = st.session_state['all_models'][prediction_model_name]
                        label_encoder = model_metrics.get('label_encoder', None)
                        
                        if label_encoder is not None:  # XGBoost
                            pred_encoded = prediction_model.predict(X_pred)[0]
                            prediction = label_encoder.inverse_transform([pred_encoded])[0]
                        else:
                            prediction = prediction_model.predict(X_pred)[0]
                    else:
                        prediction = prediction_model.predict(X_pred)[0]
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Результат предсказания")
                    
                    if hasattr(prediction_model, 'predict_proba'):
                        proba = prediction_model.predict_proba(X_pred)[0]
                        confidence = np.max(proba)
                        
                        # Получаем имена классов (с декодированием для XGBoost)
                        if 'all_models' in st.session_state and prediction_model_name in st.session_state['all_models']:
                            model_metrics = st.session_state['all_models'][prediction_model_name]
                            label_encoder = model_metrics.get('label_encoder', None)
                            
                            if label_encoder is not None:  # XGBoost
                                class_names = label_encoder.classes_
                            else:
                                class_names = prediction_model.classes_
                        else:
                            class_names = prediction_model.classes_
                        
                        # Основной результат
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.success(f"# 🏋️ {prediction}")
                            st.caption(f"Модель: {prediction_model_name}")
                            
                            # Получаем точность модели из all_models
                            if 'all_models' in st.session_state and prediction_model_name in st.session_state['all_models']:
                                model_accuracy = st.session_state['all_models'][prediction_model_name]['accuracy']
                                st.caption(f"📊 Точность модели (Accuracy): {model_accuracy:.1%} — доля правильных предсказаний на тестовой выборке")
                        
                        with col2:
                            st.metric(
                                "Уверенность (Confidence)", 
                                f"{confidence:.1%}", 
                                delta=f"{(confidence - 1/len(proba)):.1%}",
                                delta_color="off",
                                help="Вероятность предсказанного класса — насколько модель уверена в данном конкретном предсказании (от 0% до 100%)"
                            )
                            st.caption("💡 Вероятность того, что введённые данные относятся к предсказанному упражнению")
                        
                        # Таблица вероятностей
                        st.markdown("#### 📊 Вероятности по всем классам:")
                        proba_df = pd.DataFrame({
                            'Упражнение': prediction_model.classes_,
                            'Вероятность': proba,
                            'Процент': [f"{p:.2%}" for p in proba]
                        }).sort_values('Вероятность', ascending=False).reset_index(drop=True)
                        
                        # Подсвечиваем максимальную вероятность
                        def highlight_max(s):
                            is_max = s == s.max()
                            return ['background-color: lightgreen' if v else '' for v in is_max]
                        
                        styled_proba = proba_df.style.apply(highlight_max, subset=['Вероятность'])
                        st.dataframe(styled_proba, use_container_width=True, height=350)
                        
                        # График вероятностей
                        st.markdown("#### 📈 Визуализация вероятностей:")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        colors = ['lightcoral' if i != 0 else 'steelblue' 
                                 for i in range(len(proba_df))]
                        
                        bars = ax.barh(proba_df['Упражнение'], proba_df['Вероятность'], 
                                      color=colors, edgecolor='navy', linewidth=1.5)
                        ax.set_xlabel('Вероятность', fontsize=12, fontweight='bold')
                        ax.set_title(f'Распределение вероятностей ({prediction_model_name})', 
                                   fontsize=14, fontweight='bold')
                        ax.set_xlim([0, 1])
                        ax.grid(axis='x', alpha=0.3)
                        
                        # Добавление значений на столбцы
                        for i, (bar, prob) in enumerate(zip(bars, proba_df['Вероятность'])):
                            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                                   f'{prob:.1%}', va='center', fontsize=10, fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Дополнительная информация
                        with st.expander("ℹ️ Интерпретация результатов"):
                            # Получаем точность модели
                            model_accuracy = "N/A"
                            if 'all_models' in st.session_state and prediction_model_name in st.session_state['all_models']:
                                model_accuracy = f"{st.session_state['all_models'][prediction_model_name]['accuracy']:.1%}"
                            
                            st.markdown(f"""
                            **Как интерпретировать результат:**
                            
                            **📊 Два ключевых понятия:**
                            
                            1. **Уверенность (Confidence):** `{confidence:.1%}`
                               - Это вероятность того, что данные относятся именно к предсказанному классу
                               - Показывает, насколько модель уверена в **этом конкретном предсказании**
                               - Рассчитывается как максимальная вероятность среди всех классов
                               - **Для одного примера** (текущего ввода)
                            
                            2. **Точность модели (Accuracy):** `{model_accuracy}`
                               - Это общая метрика качества модели на всей тестовой выборке
                               - Показывает долю правильных предсказаний от всех примеров
                               - Рассчитывается как: (правильные предсказания) / (всего примеров)
                               - **Для всего набора данных**
                            
                            **Разница:** Точность — это общая характеристика модели, а уверенность — это её уверенность в конкретном случае.
                            
                            ---
                            
                            **Текущее предсказание:**
                            - **Предсказанное упражнение:** `{prediction}` — класс с максимальной вероятностью
                            - **Уверенность:** `{confidence:.1%}` — вероятность этого класса
                            - **Порог надёжности:** Обычно >70% считается надёжным предсказанием
                            
                            **Статус уверенности:** {'✅ Высокая уверенность (>70%) — предсказание надёжно' if confidence > 0.7 
                                               else '⚠️ Средняя уверенность (50-70%) — предсказание вероятно верно' if confidence > 0.5 
                                               else '❌ Низкая уверенность (<50%) — предсказание сомнительно'}
                            
                            **Что это означает:**
                            {'Модель с высокой уверенностью определила упражнение. Можно доверять результату.' if confidence > 0.7
                             else 'Модель сомневается между несколькими классами. Проверьте другие вероятности.' if confidence > 0.5
                             else 'Модель не может уверенно определить упражнение. Возможно, данные выходят за рамки обучающей выборки.'}
                            """)
                        
                        # Сравнение с другими моделями (если доступно)
                        if 'all_models' in st.session_state and len(st.session_state['all_models']) > 1:
                            with st.expander("🔄 Сравнить с другими моделями"):
                                st.markdown("**Предсказания всех обученных моделей на введённых данных:**")
                                st.caption("Сравнение показывает, как разные модели классифицируют один и тот же пример")
                                
                                comparison_results = []
                                for model_name, model_metrics in st.session_state['all_models'].items():
                                    model_obj = model_metrics['model']
                                    label_encoder_m = model_metrics.get('label_encoder', None)
                                    
                                    # Предсказание с декодированием для XGBoost
                                    if label_encoder_m is not None:
                                        pred_encoded = model_obj.predict(X_pred)[0]
                                        pred = label_encoder_m.inverse_transform([pred_encoded])[0]
                                    else:
                                        pred = model_obj.predict(X_pred)[0]
                                    
                                    # Уверенность
                                    if hasattr(model_obj, 'predict_proba'):
                                        prob = model_obj.predict_proba(X_pred)[0]
                                        conf = np.max(prob)
                                    else:
                                        conf = None
                                    
                                    # Точность модели
                                    accuracy = model_metrics['accuracy']
                                    
                                    comparison_results.append({
                                        'Модель': model_name,
                                        'Точность модели': f"{accuracy:.2%}",
                                        'Предсказание': pred,
                                        'Уверенность': f"{conf:.2%}" if conf else "N/A",
                                        'Совпадает': '✅' if pred == prediction else '❌'
                                    })
                                
                                comp_df = pd.DataFrame(comparison_results)
                                
                                # Сортируем по точности модели
                                comp_df['_accuracy_sort'] = comp_df['Точность модели'].str.rstrip('%').astype(float)
                                comp_df = comp_df.sort_values('_accuracy_sort', ascending=False).drop('_accuracy_sort', axis=1)
                                
                                st.dataframe(comp_df, use_container_width=True)
                                
                                # Краткие выводы
                                matching_models = sum(1 for r in comparison_results if r['Совпадает'] == '✅')
                                total_models = len(comparison_results)
                                
                                st.info(f"""
                                **Консенсус моделей:** {matching_models} из {total_models} моделей ({matching_models/total_models*100:.0f}%) 
                                предсказали класс `{prediction}`.
                                
                                {'✅ Высокая согласованность — предсказание надёжно' if matching_models/total_models > 0.7
                                 else '⚠️ Средняя согласованность — модели расходятся во мнениях' if matching_models/total_models > 0.5
                                 else '❌ Низкая согласованность — результат сомнителен'}
                                """)
                    else:
                        # Для моделей без вероятностей
                        st.success(f"### 🎯 Предсказанное упражнение: **{prediction}**")
                        st.caption(f"Модель: {prediction_model_name}")
                        st.info("⚠️ Данная модель не поддерживает вывод вероятностей")

else:
    st.info("👆 Загрузите данные или сгенерируйте демо-датасет для начала работы")