import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedBindingDBCollector:
    def __init__(self):
        # Инициализация клиентов ChEMBL
        self.target = new_client.target
        self.activity = new_client.activity
        self.molecule = new_client.molecule
        self.mechanism = new_client.mechanism
        self.checkpoint_file = 'collection_checkpoint.json'

    def get_protein_data_from_chembl(self, target_chembl_id):
        """Получение данных о белке из ChEMBL"""
        try:
            # Получаем информацию о белке
            target = self.target.get(target_chembl_id)
            
            # Проверка на человеческий белок
            if target['organism'] != 'Homo sapiens':
                logging.warning(f"Пропуск не человеческого белка {target_chembl_id}")
                return None
                
            #компоненты белка 
            components = target.get('target_components', [])
            if not components:
                logging.warning(f"Нет данных о компонентах белка {target_chembl_id}")
                return None
                
            sequence = components[0].get('sequence', '')
            if not sequence:
                logging.warning(f"Нет последовательности для белка {target_chembl_id}")
                return None

            return {
                'id': target_chembl_id,
                'sequence': sequence,
                'name': target.get('pref_name', ''),
                'gene': components[0].get('gene_name', 'Unknown'),
                'organism': target['organism'],
                'protein_class': target.get('target_type', 'Unknown')
            }

        except Exception as e:
            logging.error(f"Ошибка при получении данных белка {target_chembl_id}: {str(e)}")
            return None

    def get_all_protein_targets_from_chembl(self):
        """Получение списка белковых мишеней из ChEMBL"""
        try:
            # список всех человеческих белковых мишеней
            targets = self.target.filter(
                target_type="SINGLE PROTEIN",
                organism="Homo sapiens",
                limit=1000 
            )
            
            
            target_ids = []
            for target in targets:
                target_id = target['target_chembl_id']
                # Проверяем наличие данных об активности
                activities = self.activity.filter(
                    target_chembl_id=target_id,
                    standard_type__in=['IC50', 'Ki'],
                    limit=1
                )
                if len(list(activities)) > 0:
                    target_ids.append(target_id)

            logging.info(f"Успешно получено {len(target_ids)} белковых мишеней из ChEMBL")
            return target_ids

        except Exception as e:
            logging.error(f"Ошибка при получении белковых мишеней из ChEMBL: {str(e)}")
            return []

    def get_inhibitors_chembl(self, target_chembl_id, limit=3):
        """Получение ингибиторов для белка из ChEMBL"""
        try:
            #механизмы действия для целевого белка
            mechanisms = self.mechanism.filter(
                target_chembl_id=target_chembl_id,
                action_type__icontains='INHIBITOR'
            )

            # активность для ингибиторов
            activities = self.activity.filter(
                target_chembl_id=target_chembl_id,
                standard_type__in=['IC50', 'Ki'],
                standard_relation__in=['=', '<', '<='],
                standard_units='nM'
            ).order_by('standard_value')[:limit]

            inhibitors = []
            for act in activities:
                if not (act.get('standard_value') and act.get('molecule_chembl_id')):
                    continue

                # информация о молекуле
                molecule = self.molecule.get(act['molecule_chembl_id'])
                if not molecule.get('molecule_structures', {}).get('canonical_smiles'):
                    continue

                inhibitors.append({
                    'smiles': molecule['molecule_structures']['canonical_smiles'],
                    'measurement_type': act['standard_type'],
                    'activity_value': float(act['standard_value']),
                    'p_activity': -np.log10(float(act['standard_value']) * 1e-9),
                    'molecule_chembl_id': act['molecule_chembl_id']
                })

            return inhibitors

        except Exception as e:
            logging.error(f"Ошибка получения ингибиторов ChEMBL для {target_chembl_id}: {str(e)}")
            return []

    def get_extended_protein_properties(self, sequence):
        """Расчет свойств белка"""
        try:
            protein = ProteinAnalysis(sequence)
            properties = {
                'protein_weight': protein.molecular_weight(),
                'protein_aromaticity': protein.aromaticity(),
                'protein_instability_index': protein.instability_index(),
                'protein_isoelectric_point': protein.isoelectric_point(),
                'protein_helix_fraction': protein.secondary_structure_fraction()[0],
                'protein_turn_fraction': protein.secondary_structure_fraction()[1],
                'protein_sheet_fraction': protein.secondary_structure_fraction()[2]
            }
            return properties
        except Exception as e:
            logging.error(f"Ошибка при анализе белка: {str(e)}")
            return {}

    def get_extended_inhibitor_properties(self, mol):
        """Расчет свойств ингибитора"""
        try:
            return {
                'inhibitor_molecular_weight': Descriptors.ExactMolWt(mol),
                'inhibitor_logp': Descriptors.MolLogP(mol),
                'inhibitor_hbd': Descriptors.NumHDonors(mol),
                'inhibitor_hba': Descriptors.NumHAcceptors(mol),
                'inhibitor_tpsa': Descriptors.TPSA(mol),
                'inhibitor_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'inhibitor_rings': Descriptors.RingCount(mol),
                'inhibitor_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'inhibitor_complexity': Descriptors.BertzCT(mol),
                'inhibitor_qed': QED.default(mol)
            }
        except Exception as e:
            logging.error(f"Ошибка при расчете свойств ингибитора: {str(e)}")
            return {}

    def collect_and_process_data(self, batch_size=10, inhibitors_per_protein=3):
        """Сбор и обработка данных"""
        data = []
        processed_count = 0

        # список белков
        protein_list = self.get_all_protein_targets_from_chembl()
        if not protein_list:
            logging.error("Не удалось получить список белков")
            return pd.DataFrame()

        # чекпоинт
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    data = checkpoint['data']
                    processed_count = checkpoint['processed_count']
                    logging.info(f"Загружен чекпоинт: обработано {processed_count} белков")
        except Exception as e:
            logging.warning(f"Не удалось загрузить чекпоинт: {str(e)}")

        # Обработка белков
        for i in range(processed_count, len(protein_list), batch_size):
            batch_proteins = protein_list[i:i + batch_size]
            
            for protein in batch_proteins:
                #данные о белке
                protein_data = self.get_protein_data_from_chembl(protein)
                if not protein_data:
                    logging.warning(f"Пропуск белка {protein} из-за отсутствия данных")
                    continue

                # свойства белка
                protein_properties = self.get_extended_protein_properties(protein_data['sequence'])

                # ингибиторы
                inhibitors = self.get_inhibitors_chembl(protein, limit=inhibitors_per_protein)
                if not inhibitors:
                    logging.warning(f"Нет ингибиторов для белка {protein}")
                    continue

                
                for inhibitor in inhibitors:
                    mol = Chem.MolFromSmiles(inhibitor['smiles'])
                    if mol is None:
                        logging.warning(f"Неверный SMILES: {inhibitor['smiles']}")
                        continue

                    # Собираем все данные вместе
                    entry = {
                        'protein_id': protein_data['id'],
                        'protein_sequence': protein_data['sequence'],
                        'protein_name': protein_data['name'],
                        'protein_gene': protein_data['gene'],
                        'protein_organism': protein_data['organism'],
                        'protein_class': protein_data['protein_class']
                    }
                    entry.update(protein_properties)
                    entry.update({
                        'inhibitor_smiles': inhibitor['smiles'],
                        'inhibitor_chembl_id': inhibitor['molecule_chembl_id'],
                        'activity_type': inhibitor['measurement_type'],
                        'activity_value': inhibitor['activity_value'],
                        'p_activity': inhibitor['p_activity']
                    })
                    inhibitor_properties = self.get_extended_inhibitor_properties(mol)
                    entry.update(inhibitor_properties)
                    data.append(entry)

                processed_count += 1
                logging.info(f"Обработано белков: {processed_count}/{len(protein_list)}")

            # Сохранение чекпоинта после каждой группы чтобы в случае ошибки не потерять данные 
            self.save_checkpoint(data, processed_count)

        return pd.DataFrame(data)

    def save_checkpoint(self, data, processed_count):
        """Сохранение промежуточных результатов"""
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'processed_count': processed_count,
                'data': data
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            logging.info(f"Сохранен чекпоинт: обработано {processed_count} белков")
        except Exception as e:
            logging.error(f"Ошибка сохранения чекпоинта: {str(e)}")

def main():
    collector = EnhancedBindingDBCollector()
    df = collector.collect_and_process_data(batch_size=10, inhibitors_per_protein=3)
    
    if df.empty:
        logging.error("Нет собранных данных для обучения.")
        return

    
    feature_columns = [col for col in df.columns if col.startswith(('protein_', 'inhibitor_')) 
                      and col not in ['protein_id', 'protein_sequence', 'protein_name', 
                                    'protein_gene', 'protein_organism', 'protein_class', 
                                    'inhibitor_smiles', 'inhibitor_chembl_id']]
    
    X = df[feature_columns]
    y = df['p_activity']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Стандартизация признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Обучение моделей
    models = {
        'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        logging.info(f"Модель {name}:")
        logging.info(f"MAE: {mae:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()