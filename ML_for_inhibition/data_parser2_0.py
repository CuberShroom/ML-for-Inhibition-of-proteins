from asyncio import as_completed
from chembl_webresource_client.new_client import new_client
import pubchempy as pcp
import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings
warnings.filterwarnings('ignore')
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import json
from datetime import datetime
import random
from Bio.SeqUtils.ProtParam import ProteinAnalysis


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='api_connection.log'
)

class ProteinInhibitorCollector:
    def __init__(self):
        
        self.session = requests.Session()
        retries = Retry(
            total=10,  
            backoff_factor=2,  
            status_forcelist=[500, 502, 503, 504, 429],  
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.timeout = (30, 300)  # (connect timeout, read timeout)
        
       
        self.api_urls = {
            'primary': 'https://www.ebi.ac.uk/chembl/api/data/status',
            'backup': 'https://www.ebi.ac.uk/chembl/api/data/status',
            'fallback': 'https://www.ebi.ac.uk/chembl/api/data/status'
        }
        
        # Проверка соединения перед инициализацией
        if self.check_connection():
            try:
                self.target = new_client.target
                self.compound = new_client.molecule
                self.activity = new_client.activity
                print("Успешное подключение к ChEMBL API")
            except Exception as e:
                logging.error(f"Ошибка при инициализации API: {str(e)}")
                raise
        else:
            raise ConnectionError("Не удалось установить соединение с ChEMBL API")

    def check_connection(self):
        """Проверка соединения с API с несколькими попытками"""
        for name, url in self.api_urls.items():
            try:
                print(f"Попытка подключения к {name} URL...")
                response = self.session.get(
                    url,
                    timeout=(30, 300),  
                    headers={'User-Agent': 'Mozilla/5.0'} 
                )
                if response.status_code == 200:
                    print(f"✓ Соединение с ChEMBL установлено через {name} URL")
                    return True
                else:
                    print(f"✗ Ошибка соединения с {name} URL: статус {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"✗ Таймаут при подключении к {name} URL")
                logging.warning(f"Таймаут при подключении к {name} URL")
                continue
            except requests.exceptions.RequestException as e:
                print(f"✗ Ошибка при подключении к {name} URL: {str(e)}")
                logging.error(f"Ошибка соединения с {name} URL: {str(e)}")
                continue
        return False

    def safe_api_call(self, func, *args, max_retries=5, delay=10):
        """Безопасный вызов API с увеличенным количеством попыток"""
        for attempt in range(max_retries):
            try:
                return func(*args)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logging.error(f"Ошибка API после {max_retries} попыток: {str(e)}")
                    raise
                wait_time = delay * (2 ** attempt)  # Экспоненциальное увеличение времени ожидания
                print(f"Попытка {attempt + 1} не удалась. Ожидание {wait_time} секунд...")
                time.sleep(wait_time)
                continue

    def get_all_protein_targets(self):
        """Получение всех белковых мишеней из ChEMBL с таймаутом и обработкой ошибок"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"Попытка {attempt + 1} получения белковых мишеней...")
                
                
                targets = self.target.filter(
                    target_type="SINGLE PROTEIN",
                    organism="Homo sapiens"
                ).only(['target_chembl_id', 'pref_name', 'target_type', 'organism'])
                
                
                targets_list = list(targets)
                
                print(f"Успешно получено {len(targets_list)} белковых мишеней")
                return targets_list
                
            except requests.exceptions.Timeout:
                print(f"Таймаут при попытке {attempt + 1}")
                if attempt < max_retries - 1:
                    print(f"Ожидание {retry_delay} секунд перед следующей попыткой...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Превышено максимальное количество попыток")
                    return []
                
            except Exception as e:
                print(f"Ошибка при попытке {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Ожидание {retry_delay} секунд перед следующей попыткой...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Превышено максимальное количество попыток")
                    return []

    def get_uniprot_data(self, protein_name):
        """Получение данных о белке из UniProt с обработкой ошибок"""
        try:
            params = {
                'query': protein_name,
                'format': 'json'
            }
            response = self.session.get(self.uniprot_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    return data['results'][0]
            return None
        except requests.exceptions.Timeout:
            print(f"Таймаут при запросе к UniProt для {protein_name}")
            logging.warning(f"Таймаут UniProt: {protein_name}")
            return None
        except Exception as e:
            logging.error(f"Ошибка при получении данных UniProt для {protein_name}: {str(e)}")
            return None

    def get_compound_properties(self, smiles):
        """Расчет молекулярных свойств соединения"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            properties = {
                'molecular_weight': Descriptors.ExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'fingerprint': list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024).ToBitString())
            }
            return properties
        except Exception as e:
            logging.error(f"Ошибка при расчете свойств для SMILES {smiles}: {str(e)}")
            return None

    def get_pubchem_data(self, compound_id):
        """Получение данных о соединении из PubChem"""
        try:
            compound = pcp.Compound.from_cid(compound_id)
            return {
                'iupac_name': compound.iupac_name,
                'molecular_formula': compound.molecular_formula,
                'canonical_smiles': compound.canonical_smiles,
                'xlogp': compound.xlogp,
                'complexity': compound.complexity
            }
        except:
            return None

    def collect_inhibitor_data(self, target_chembl_id):
        """Сбор данных об ингибиторах для конкретной мишени"""
        try:
            activities = self.activity.filter(
                target_chembl_id=target_chembl_id,
                standard_type__in=['IC50', 'Ki', 'EC50', 'Kd'],
                standard_relation__in=['=', '<', '<=', '>', '>=']
            )

            inhibitor_data = []
            for act in activities:
                if not (act.get('standard_value') and act.get('canonical_smiles')):
                    continue

                compound_data = {
                    'molecule_chembl_id': act['molecule_chembl_id'],
                    'smiles': act['canonical_smiles'],
                    'standard_type': act['standard_type'],
                    'standard_value': act['standard_value'],
                    'standard_units': act['standard_units'],
                }

                # Добавление молекулярных свойств
                properties = self.get_compound_properties(act['canonical_smiles'])
                if properties:
                    compound_data.update(properties)

                inhibitor_data.append(compound_data)

            return inhibitor_data
        except Exception as e:
            logging.error(f"Ошибка при сборе данных об ингибиторах для {target_chembl_id}: {str(e)}")
            return []

    def process_target(self, target):
        """Обработка одной белковой мишени"""
        try:
            target_data = {
                'target_chembl_id': target['target_chembl_id'],
                'protein_name': target['pref_name'],
                'organism': target['organism'],
                'target_type': target['target_type']
            }

            # Получение данных UniProt
            uniprot_data = self.get_uniprot_data(target['pref_name'])
            if uniprot_data:
                target_data.update({
                    'uniprot_id': uniprot_data.get('primaryAccession'),
                    'protein_function': uniprot_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value')
                })

            # Получение данных об ингибиторах
            inhibitors = self.collect_inhibitor_data(target['target_chembl_id'])
            
            # Создание записей для каждого ингибитора
            records = []
            for inhibitor in inhibitors:
                record = target_data.copy()
                record.update(inhibitor)
                records.append(record)

            return records
        except Exception as e:
            logging.error(f"Ошибка при обработке мишени {target.get('target_chembl_id')}: {str(e)}")
            return []

    def collect_all_data(self):
        """Основной метод для сбора всех данных с промежуточным сохранением"""
        all_data = []
        checkpoint_file = 'protein_inhibitors_checkpoint.json'
        processed_targets = set()
        
        # Попытка загрузить предыдущий прогресс
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    all_data = checkpoint['data']
                    processed_targets = set(checkpoint['processed_targets'])
                    print(f"Загружено {len(all_data)} записей из предыдущей сессии")
        except Exception as e:
            logging.warning(f"Не удалось загрузить предыдущий прогресс: {str(e)}")
        
        try:
            targets = self.get_all_protein_targets()
            print(f"Найдено {len(targets)} белковых мишеней")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                
                for target in targets:
                    target_id = target['target_chembl_id']
                    if target_id in processed_targets:
                        continue
                        
                    futures.append(executor.submit(self.process_target, target))
                    
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        records = future.result()
                        if records:
                            all_data.extend(records)
                            processed_targets.add(records[0]['target_chembl_id'])
                            
                            # Сохранение прогресса каждые 100 новых записей
                            if len(all_data) % 100 == 0:
                                self.save_checkpoint(all_data, processed_targets, checkpoint_file)
                                self.save_current_dataset(all_data)
                                
                    except Exception as e:
                        logging.error(f"Ошибка при обработке мишени: {str(e)}")
                        # Сохраняем текущий прогресс даже при ошибке
                        self.save_checkpoint(all_data, processed_targets, checkpoint_file)
                        self.save_current_dataset(all_data)
        
        except Exception as e:
            logging.error(f"Критическая ошибка при сборе данных: {str(e)}")
        finally:
            # Финальное сохранение всех собранных данных
            if all_data:
                self.save_current_dataset(all_data)
                print(f"Сохранено {len(all_data)} записей")
        
        return pd.DataFrame(all_data) if all_data else pd.DataFrame()

    def save_checkpoint(self, data, processed_targets, checkpoint_file):
        """Сохранение контрольной точки"""
        try:
            checkpoint = {
                'data': data,
                'processed_targets': list(processed_targets),
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
        except Exception as e:
            logging.error(f"Ошибка при сохранении контрольной точки: {str(e)}")

    def save_current_dataset(self, data):
        """Сохранение текущего состояния датасета"""
        try:
            df = pd.DataFrame(data)
            
            # Конвертация значений в логарифмическую шкалу
            df['log_value'] = df['standard_value'].apply(
                lambda x: -np.log10(x * 1e-9) if x and x > 0 else None
            )
            
            # Сохранение в разных форматах для надежности
            df.to_csv('protein_inhibitors_current.csv', index=False)
            df.to_pickle('protein_inhibitors_current.pkl')
            
            # Сохранение статистики
            stats = {
                'total_proteins': df['target_chembl_id'].nunique(),
                'total_inhibitors': len(df),
                'last_update': datetime.now().isoformat(),
                'status': 'in_progress'
            }
            with open('collection_stats.json', 'w') as f:
                json.dump(stats, f, indent=4)
            
        except Exception as e:
            logging.error(f"Ошибка при сохранении текущего датасета: {str(e)}")

    def collect_structured_data(self, max_proteins=200, inhibitors_per_protein=3):
        """Сбор структурированных данных о белках и их ингибиторах"""
        structured_data = []
        processed_count = 0
        
        try:
            print("Начало сбора данных о белках...")
            targets = self.get_all_protein_targets()
            
            if not targets:
                print("Не удалось получить данные о белках")
                return pd.DataFrame()
            
            for target in tqdm(targets[:max_proteins], desc="Обработка белков"):
                try:
                    # Базовая информация о белке из ChEMBL
                    protein_info = {
                        'protein_chembl_id': target['target_chembl_id'],
                        'protein_name': target['pref_name'],
                        'protein_type': target['target_type'],
                        'organism': target['organism']
                    }
                    
                    # Получение данных из UniProt
                    uniprot_data = self.get_uniprot_data(target['pref_name'])
                    if uniprot_data:
                        protein_info.update({
                            'uniprot_id': uniprot_data.get('primaryAccession'),
                            'protein_sequence': uniprot_data.get('sequence', {}).get('value'),
                            'protein_mass': uniprot_data.get('mass'),
                            'protein_function': uniprot_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value'),
                            'protein_gene_name': uniprot_data.get('genes', [{}])[0].get('name', {}).get('value'),
                            'protein_subcellular_location': str(uniprot_data.get('subcellularLocations', [])),
                            'protein_families': str(uniprot_data.get('families', [])),
                            'protein_domains': str([feature.get('description') for feature in uniprot_data.get('features', []) if feature.get('type') == 'DOMAIN'])
                        })
                    
                    # Получение данных из PDB
                    pdb_id = self.get_pdb_id(target['pref_name'])
                    if pdb_id:
                        pdb_data = self.get_pdb_data(pdb_id)
                        if pdb_data:
                            protein_info.update({
                                'pdb_id': pdb_id,
                                'structure_method': pdb_data.get('structure_method'),
                                'resolution': pdb_data.get('resolution'),
                                'structure_weight': pdb_data.get('weight'),
                                'release_date': pdb_data.get('release_date')
                            })
                    
                    # Анализ последовательности белка
                    if protein_info.get('protein_sequence'):
                        try:
                            analysis = ProteinAnalysis(protein_info['protein_sequence'])
                            protein_info.update({
                                'protein_molecular_weight': analysis.molecular_weight(),
                                'protein_aromaticity': analysis.aromaticity(),
                                'protein_instability_index': analysis.instability_index(),
                                'protein_isoelectric_point': analysis.isoelectric_point(),
                                'protein_gravy': analysis.gravy(),
                                'protein_secondary_structure': str(analysis.secondary_structure_fraction())
                            })
                        except Exception as e:
                            logging.warning(f"Ошибка при анализе последовательности белка {target['pref_name']}: {str(e)}")
                    
                    # Получение ингибиторов
                    inhibitors = self.collect_inhibitor_data(target['target_chembl_id'])
                    if inhibitors:
                        # Ограничение количества ингибиторов
                        for inhibitor in inhibitors[:inhibitors_per_protein]:
                            record = protein_info.copy()
                            record.update(inhibitor)
                            structured_data.append(record)
                    
                    processed_count += 1
                    
                    # Промежуточное сохранение каждые 20 белков
                    if processed_count % 20 == 0:
                        self.save_checkpoint(structured_data, f'protein_inhibitors_temp_{processed_count}.csv')
                        logging.info(f"Сохранен промежуточный файл для {processed_count} белков")
                    
                except Exception as e:
                    logging.error(f"Ошибка при обработке белка {target['pref_name']}: {str(e)}")
                    continue
                
                if processed_count >= max_proteins:
                    break
                
        except Exception as e:
            logging.error(f"Критическая ошибка: {str(e)}")
        finally:
            # Сохранение финального датасета
            if structured_data:
                df = pd.DataFrame(structured_data)
                df.to_csv('enriched_protein_inhibitors.csv', index=False)
                
                # Сохранение статистики
                stats = {
                    'total_proteins': len(set(df['protein_chembl_id'])),
                    'total_inhibitors': len(df),
                    'average_inhibitors_per_protein': len(df) / len(set(df['protein_chembl_id'])),
                    'collection_date': datetime.now().isoformat()
                }
                
                with open('collection_stats.json', 'w') as f:
                    json.dump(stats, f, indent=4)
                
                print(f"\nСобрано:\n- Белков: {stats['total_proteins']}")
                print(f"- Ингибиторов: {stats['total_inhibitors']}")
                print(f"- Среднее число ингибиторов на белок: {stats['average_inhibitors_per_protein']:.2f}")
                
                return df
            
            return pd.DataFrame()

    def get_pdb_id(self, protein_name):
        """Получение PDB ID для белка"""
        try:
            url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={{'query': {{'type': 'terminal','service': 'text','parameters': {{'value': '{protein_name}'}},'node_id': 0}}}}"
            response = self.session.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get('result_set'):
                    return data['result_set'][0]['identifier']
            return None
        except Exception as e:
            logging.error(f"Ошибка при получении PDB ID для {protein_name}: {str(e)}")
            return None

    def get_pdb_data(self, pdb_id):
        """Получение данных о структуре белка из PDB"""
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = self.session.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logging.error(f"Ошибка при получении данных PDB для {pdb_id}: {str(e)}")
            return None

    def save_checkpoint(self, data, filename):
        """Сохранение промежуточных результатов"""
        try:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logging.info(f"Сохранен промежуточный файл: {filename}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении промежуточного файла: {str(e)}")

def main():
    try:
        collector = ProteinInhibitorCollector()
        enriched_data = collector.collect_structured_data(max_proteins=200, inhibitors_per_protein=3)
        print("Сбор данных завершен успешно")
    except Exception as e:
        print(f"Ошибка при выполнении программы: {str(e)}")

if __name__ == "__main__":
    main()

