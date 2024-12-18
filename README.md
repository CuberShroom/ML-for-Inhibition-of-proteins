# ML-for-Inhibition-of-proteins
This algorithm is based on three machine-learning models that are employed to inhibit proteins using IC50 and Ki values. It was developed as a part of a scientific contest in infochemistry, and it compiles a database of protein and inhibitor information from the ChEMBL repository, saving it in the CSV format.

Subsequently, the three models — CatBoost, LightGBM, and XGBoost — are trained using the accumulated data, and the most effective model is chosen using scikit-learn. Lastly, graphs are produced from the chosen model.
