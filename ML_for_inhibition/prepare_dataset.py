import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetPreprocessor:
    def __init__(self, input_file):
        self.data = pd.read_csv(input_file)
        self.setup_logging()
       
        logging.info(f"Доступные колонки в датасете: {self.data.columns.tolist()}")
        
    def setup_logging(self):
        logging.basicConfig(
            filename=f'dataset_preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def preprocess_dataset(self):
        """Предобработка датасета для улучшения качества модели"""
        logging.info(f"Начало обработки датасета. Исходный размер: {self.data.shape}")
        
        #удаление дубликатов
        self.data = self.data.drop_duplicates()
        
        #Нормализация значений активности
        self.normalize_activity_values()
        
        #Удаление вбросов по активности
        self.remove_activity_outliers()
        
        #Кодирование категориальных признаков
        self.encode_categorical_features()
        
        #Создание дополнительных признаков
        self.create_additional_features()
        
        #Фильтрация данных
        self.filter_data()
        
        logging.info(f"Обработка завершена. Итоговый размер: {self.data.shape}")
        return self.data
    
    def normalize_activity_values(self):
        """Нормализация значений активности с учетом типа и единиц измерения"""
        #Преобразование всех значений к единой шкале
        def convert_to_nm(row):
            value = row['inhibitor_activity_value']
            units = row['inhibitor_units']
            if pd.isnull(units):
                logging.warning(f"Отсутствуют единицы измерения для значения активности: {value}")
                return value
            if units == 'uM':
                return value * 1000
            elif units == 'mM':
                return value * 1000000
            return value
        
        # Проверяем наличие необходимых колонок
        required_columns = ['inhibitor_activity_value', 'inhibitor_units']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logging.error(f"Отсутствуют необходимые колонки: {missing_columns}")
            raise KeyError(f"Отсутствуют колонки: {missing_columns}")
        
        # Проверяем данные 
        unique_units = self.data['inhibitor_units'].unique()
        logging.info(f"Уникальные единицы измерения в данных: {unique_units}")
        
        self.data['normalized_activity'] = self.data.apply(convert_to_nm, axis=1)
        self.data['log_activity'] = np.log10(self.data['normalized_activity'])
        
        logging.info("Значения активности нормализованы и логарифмированы")
        
        # Визуализация 
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.data['inhibitor_activity_value'], bins=50, kde=True)
        plt.title('Распределение y_raw')
        
        plt.subplot(1, 2, 2)
        sns.histplot(self.data['log_activity'], bins=50, kde=True)
        plt.title('Распределение y после трансформации')
        plt.tight_layout()
        plt.savefig('y_distribution.png')
        plt.close()
        
        # Проверка на пропущенные значения
        if self.data.isnull().sum().sum() > 0:
            logging.error("В данных найдены пропущенные значения.")
            self.data = self.data.dropna()
            logging.info("Пропущенные значения удалены.")
    
    def remove_activity_outliers(self):
        """Удаление выбросов по значениям активности"""
        Q1 = self.data['log_activity'].quantile(0.25)
        Q3 = self.data['log_activity'].quantile(0.75)
        IQR = Q3 - Q1
        
        mask = (
            (self.data['log_activity'] >= Q1 - 1.5 * IQR) & 
            (self.data['log_activity'] <= Q3 + 1.5 * IQR)
        )
        
        initial_length = len(self.data)
        self.data = self.data[mask]
        removed = initial_length - len(self.data)
        logging.info(f"Удалены выбросы по активности. Удалено записей: {removed}. Осталось записей: {len(self.data)}")
    
    def encode_categorical_features(self):
        """Кодирование категориальных признаков"""
        categorical_columns = ['protein_type', 'organism', 'inhibitor_activity_type']
        
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
            else:
                logging.warning(f"Категориальная колонка '{col}' отсутствует в данных.")
        
        logging.info("Категориальные признаки закодированы")
    
    def create_additional_features(self):
        """Создание дополнительных признаков"""
        # Отношение молекулярной массы к активности
        self.data['mw_activity_ratio'] = self.data['inhibitor_molecular_weight'] / self.data['normalized_activity']
        
        # Эффективность лиганда 
        self.data['ligand_efficiency'] = -self.data['log_activity'] / self.data['inhibitor_molecular_weight']
        
        # Липофильная эффективность 
        self.data['lip_efficiency'] = -self.data['log_activity'] - self.data['inhibitor_logp']
        
        logging.info("Созданы дополнительные признаки")
    
    def filter_data(self):
        """Фильтрация данных по заданным критериям"""
        # Фильтрация по молекулярной массе 
        initial_length = len(self.data)
        self.data = self.data[self.data['inhibitor_molecular_weight'] <= 500]
        filtered_mw = initial_length - len(self.data)
        logging.info(f"Фильтрация по молекулярной массе: удалено записей: {filtered_mw}")
        
        # Фильтрация по logP 
        initial_length = len(self.data)
        self.data = self.data[self.data['inhibitor_logp'] <= 5]
        filtered_logp = initial_length - len(self.data)
        logging.info(f"Фильтрация по logP: удалено записей: {filtered_logp}")
        
        # Фильтрация по количеству доноров водородных связей
        initial_length = len(self.data)
        self.data = self.data[self.data['inhibitor_hbd'] <= 5]
        filtered_hbd = initial_length - len(self.data)
        logging.info(f"Фильтрация по hbd: удалено записей: {filtered_hbd}")
        
        # Фильтрация по количеству акцепторов водородных связей
        initial_length = len(self.data)
        self.data = self.data[self.data['inhibitor_hba'] <= 10]
        filtered_hba = initial_length - len(self.data)
        logging.info(f"Фильтрация по hba: удалено записей: {filtered_hba}")
        
        logging.info("Данные отфильтрованы по правилам Липински")
    
    def save_dataset(self, output_file):
        """Сохранение обработанного датасета"""
        self.data.to_csv(output_file, index=False)
        logging.info(f"Датасет сохранен в {output_file}")
        
        # Сохранение статистики
        stats = {
            'total_compounds': len(self.data),
            'unique_proteins': self.data['protein_chembl_id'].nunique(),
            'activity_range': {
                'min': self.data['log_activity'].min(),
                'max': self.data['log_activity'].max(),
                'mean': self.data['log_activity'].mean()
            },
            'molecular_weight_range': {
                'min': self.data['inhibitor_molecular_weight'].min(),
                'max': self.data['inhibitor_molecular_weight'].max(),
                'mean': self.data['inhibitor_molecular_weight'].mean()
            }
        }
        
        pd.DataFrame([stats]).to_json('dataset_stats.json', orient='records', indent=4)
        logging.info("Статистика датасета сохранена в 'dataset_stats.json'")
        
def main():
    preprocessor = DatasetPreprocessor('protein_inhibitors_structured.csv')
    
    logging.info("\nПервые строки датасета:")
    logging.info(preprocessor.data.head())
    
    processed_data = preprocessor.preprocess_dataset()
    preprocessor.save_dataset('processed_inhibitors_data.csv')

if __name__ == "__main__":
    main() 