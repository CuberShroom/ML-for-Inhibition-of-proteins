import os
import pickle
import logging
from datetime import datetime, timedelta

class Cache:
    """Класс для кэширования результатов запросов к API"""
    def __init__(self, cache_dir='cache', expiration_days=7):
        self.cache_dir = cache_dir
        self.expiration_days = expiration_days
        
        # Создаем директорию для кэша, если её нет
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, key):
        """Получает путь к файлу кэша"""
        return os.path.join(self.cache_dir, f"{hash(key)}.cache")
    
    def get(self, key):
        """Получает данные из кэша"""
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Проверяем срок действия кэша
                if datetime.now() - cached_data['timestamp'] < timedelta(days=self.expiration_days):
                    return cached_data['data']
                
                # Удаляем устаревший кэш
                os.remove(cache_path)
            except Exception as e:
                logging.error(f"Ошибка чтения кэша: {str(e)}")
        
        return None
    
    def set(self, key, data):
        """Сохраняет данные в кэш"""
        cache_path = self._get_cache_path(key)
        
        try:
            cached_data = {
                'timestamp': datetime.now(),
                'data': data
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            logging.error(f"Ошибка записи кэша: {str(e)}") 