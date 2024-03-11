# Copyright 2024 Intesa SanPaolo S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import os
import time
from dtor import DTOR
from alibi.explainers.anchors.anchor_tabular import AnchorTabular
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib

# Set permissions and backend
os.umask(0o02)
matplotlib.use('agg')

# List of datasets
lst_datasets = [
    '/glass+identification/',
    '/ionosphere/',
    '/lymphography/',
    '/musk+version+2/',
    '/breast+cancer+wisconsin+diagnostic/',
    '/arrhythmia/'
]

# Loop through datasets
for main_path in lst_datasets:
    print(main_path)
    # Load dataset
    dtf = pd.read_csv(main_path + "data.data", header=None, index_col=None).reset_index(drop=True)
    if dtf.shape[0] == dtf[0].nunique():
        del dtf[0]
    y_original = dtf.iloc[:, -1]
    n_last_col = dtf.shape[1]
    del dtf[dtf.columns[-1]]
    dtf.columns = ['feature_' + str(i) for i in dtf.columns] 
    dtf = dtf.replace('?', -999)
    for i in dtf.columns:
        if dtf[i].dtype == 'float':
            dtf[i] = dtf[i].astype('float32')
        elif dtf[i].dtype == 'int':
            pass
        else:
            try:
                dtf[i] = dtf[i].astype('int64')
            except:
                label_encoder = LabelEncoder()
                dtf[i] = label_encoder.fit_transform(dtf[i])
    X_train, X_test = train_test_split(dtf, test_size=50, random_state=42)
    X_train = X_train.sample(min(5000, X_train.shape[0]))

    # Anomaly Detection (AD) model
    lst_models = [
        (IsolationForest(random_state=42, n_jobs=-1, contamination=0.05).fit(X_train),"if"),
        (OneClassSVM(kernel="rbf").fit(X_train),"svm"),
        (GaussianMixture(random_state=42), "gmm")
    ]

    for clf_ad, clf_name in lst_models:
        if clf_name == "gmm":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            clf_ad.fit(X_train_scaled)
            y_train_score = clf_ad.score_samples(X_train_scaled)
            y_test_score = clf_ad.score_samples(X_test_scaled)
            density_threshold = np.percentile(y_train_score, 5)
            y_train = [-1 if elem < density_threshold else 1 for elem in y_train_score]
            y_test = [-1 if elem < density_threshold else 1 for elem in y_test_score]
            th = density_threshold
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=X_test.index)
            dct_categorical = {}
            def predict_for_anchor(X):
                X_qt = pd.DataFrame(X, columns=X_train.columns)
                score_samples = clf_ad.score_samples(X_qt)
                return np.array([-1 if elem < density_threshold else 1 for elem in score_samples])
        else:
            th = clf_ad.offset_
            if isinstance(th, float):
                pass
            else:
                th = th[0]
            y_train = clf_ad.predict(X_train)
            y_test = clf_ad.predict(X_test)
            y_train = 1 * (y_train == -1)
            y_test = 1 * (y_test == -1)
            y_train_score = clf_ad.score_samples(X_train)
            y_test_score = clf_ad.score_samples(X_test)
            dct_categorical = {}
            def predict_for_anchor(X):
                X_qt = pd.DataFrame(X, columns=X_train.columns)
                return clf_ad.predict(X_qt)

        explainer = AnchorTabular(
            predictor=predict_for_anchor,
            feature_names=X_train.columns,
            categorical_names=dct_categorical
        )

        explainer.fit(X_train.values)

        dct_anchor = {}
        for ID in X_test.index:
            print(f"Anchor computation for id test set #: {str(ID)}")
            np.random.seed(1)  
            start = time.time()
            exp = explainer.explain(X_test.loc[ID].values, threshold=0.5, batch_size=100,
                                     delta=0.1, tau=0.15, beam_size=4)
            end = time.time()
            import re
            rules = [re.sub(r"\s=\s", " == ", r) for r in exp.anchor]
            rule_not_fulfilled = []
            for rule in rules:
                if X_test.loc[ID:].astype("float64").query(rule).shape[0] == 0:
                    rule_not_fulfilled.append(rule)
            if clf_name == "gmm":
                predict_hard = -1 if clf_ad.score_samples(X_test.loc[ID].values.reshape(1, -1)) < th else 1
            else:
                predict_hard = clf_ad.predict(X_test.loc[ID].values.reshape(1, -1))[0]
            dct_anchor[str(ID)] = {
                'score_AD': clf_ad.score_samples(X_test.loc[ID].values.reshape(1, -1)),
                'prediction_hard_AD': predict_hard,
                'prediction_hard_surrogate': exp.raw['prediction'][0],
                'precision': exp.precision,
                'coverage': exp.coverage,
                'Prediction AD': clf_ad.score_samples(X_test.loc[ID].values.reshape(1, -1))[0],
                'rule_length': len(exp.anchor),
                'rule': " AND ".join(exp.anchor),
                'Rules not fulfilled': ' AND '.join(rule_not_fulfilled),
                'Execution_time': end - start}
        pd.DataFrame(dct_anchor).T.to_html(os.path.join(main_path, f"expl_id_anchor_{clf_name}.html"),
                                            float_format=lambda x: f"{x:.4}",
                                            justify="center")
        pd.DataFrame(dct_anchor).T.to_csv(os.path.join(main_path, f"expl_id_anchor_{clf_name}.csv"))

        # DTOR
        dtol = DTOR(X_train, y_train_score, clf_ad)
        X_expl = X_test
        y_pred_expl = clf_ad.score_samples(X_expl)
        print("label to explain: ", y_pred_expl)
        dtol_config = {'max_depth': 8, 'splitter': 'best', 'min_impurity_decrease': 0.00001, 'beta': 0.1}
        try:
            output = dtol.explain_instances_with_precision(X_expl, y_pred_expl,
                                                           N_gen=100,
                                                           random_state=5, th=th,
                                                           **dtol_config)
        except:
            dtol_config = {'max_depth': 8, 'splitter': 'best', 'min_impurity_decrease': 0.0, 'beta': 0.1}
            output = dtol.explain_instances_with_precision(X_expl, y_pred_expl,
                                                           N_gen=100,
                                                           random_state=5, th=th,
                                                           **dtol_config)
        pd.DataFrame(output).T.to_html(os.path.join(main_path, f"expl_id_dtolr_{clf_name}.html"),
                                       float_format=lambda x: f"{x:.2}",
                                       justify="center")
        pd.DataFrame(output).T.to_csv(os.path.join(main_path, f"expl_id_dtolr_{clf_name}.csv"))

# Make the result tables
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

counter = 0
df_table_final = pd.DataFrame()
for main_path in lst_datasets:
    print(main_path)
    dataset = main_path.split('/')[-2]
    dct_performance = {}
    for str_model in ['if', 'gmm', 'svm']:
        if os.path.exists(main_path + 'expl_id_anchor_' + str_model + '.csv') and \
                os.path.exists(main_path + 'expl_id_dtolr_' + str_model + '.csv'):
            df_anchor = pd.read_csv(main_path + 'expl_id_anchor_' + str_model + '.csv', index_col=[0])
            df_dtor = pd.read_csv(main_path + 'expl_id_dtolr_' + str_model + '.csv', index_col=[0])

            dct_performance[str_model] = {'Execution time':
                                               {'Anchor': f"{df_anchor.loc[df_anchor['rule'].notna(), 'Execution_time'].mean():.2f} ({df_anchor.loc[df_anchor['rule'].notna(), 'Execution_time'].max():.2f})",
                                                'DTOR': f"{df_dtor.loc[df_dtor['rule'].notna(), 'Execution_time'].mean():.2f} ({df_dtor.loc[df_dtor['rule'].notna(), 'Execution_time'].max():.2f})",
                                                },
                                           'Precision':
                                               {'Anchor': f"{df_anchor.loc[df_anchor['rule'].notna(), 'precision'].astype('float64').mean():.2f} $\pm$ {df_anchor.loc[df_anchor['rule'].notna(), 'precision'].astype('float64').std():.2f}",
                                                'DTOR': f"{df_dtor.loc[df_dtor['rule'].notna(), 'precision'].astype('float64').mean():.2f} $\pm$ {df_dtor.loc[df_dtor['rule'].notna(), 'precision'].astype('float64').std():.2f}",
                                                },
                                           'Coverage':
                                               {'Anchor': f"{df_anchor.loc[df_anchor['rule'].notna(), 'coverage'].mean():.2f} $\pm$ {df_anchor.loc[df_anchor['rule'].notna(), 'coverage'].std():.2f}",
                                                'DTOR': f"{df_dtor.loc[df_dtor['rule'].notna(), 'coverage'].mean():.2f} $\pm$ {df_dtor.loc[df_dtor['rule'].notna(), 'coverage'].std():.2f}",
                                                },
                                           'Validity \%':
                                               {'Anchor': f"{int(100*(df_anchor['rule'].notna() & df_anchor['Rules not fulfilled'].isna()).mean())}",
                                                'DTOR': f"{int(100*(df_dtor['rule'].notna() & df_dtor['Rules not fulfilled'].isna()).mean())}",
                                                },
                                           'Rule length':
                                               {'Anchor': f"{df_anchor['rule_length'].mean():.2f}",
                                                'DTOR': f"{df_dtor['rule_length'].mean():.2f}",
                                                },
                                           }
        else:
            df_dtor = pd.read_csv(main_path + 'expl_id_dtolr_' + str_model + '.csv', index_col=[0])
            dct_performance[str_model] = {'Execution time':
                                               {
                                                   'Anchor': "-",
                                                   'DTOR': f"{df_dtor.loc[df_dtor['rule'].notna(), 'Execution_time'].mean():.2f} ({df_dtor.loc[df_dtor['rule'].notna(), 'Execution_time'].max():.2f})",
                                                   },
                                           'Precision':
                                               {
                                                   'Anchor': "-",
                                                   'DTOR': f"{df_dtor.loc[df_dtor['rule'].notna(), 'precision'].astype('float64').mean():.2f} $\pm$ {df_dtor.loc[df_dtor['rule'].notna(), 'precision'].astype('float64').std():.2f}",
                                                   },
                                           'Coverage':
                                               {
                                                   'Anchor': "-",
                                                   'DTOR': f"{df_dtor.loc[df_dtor['rule'].notna(), 'coverage'].mean():.2f} $\pm$ {df_dtor.loc[df_dtor['rule'].notna(), 'coverage'].std():.2f}",
                                                   },
                                           'Validity \%':
                                               {
                                                   'Anchor': "-",
                                                   'DTOR': f"{int(100 * (df_dtor['rule'].notna() & df_dtor['Rules not fulfilled'].isna()).mean())}",
                                                   },
                                           'Rule length':
                                               {'Anchor': "-",
                                                'DTOR': f"{df_dtor['rule_length'].mean():.2f}",
                                                },
                                           }
        lst_metrics = list(dct_performance[str_model].keys())

    reformed_dict = {}
    for outerKey, innerDict in dct_performance.items():
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey, innerKey)] = values

    # Multiindex dataframe
    df_table = pd.DataFrame(reformed_dict).T
    df_table.columns = pd.MultiIndex.from_product([[dataset], ['Anchor', 'DTOR']])
    df_table.to_csv(main_path + 'result_dataset_' + dataset + '.csv')
    df_table.to_html(main_path + 'result_dataset_' + dataset + '.html')
    if counter == 0:
        df_table_final = df_table
        counter += 1
    else:
        df_table_final = pd.concat([df_table_final, df_table], axis=1)

# Consolidated result table
df_table_final_reformat = pd.DataFrame({}, index=pd.MultiIndex.from_product(
    [[main_path.split('/')[-2] for main_path in lst_datasets], lst_metrics]),
    columns=pd.MultiIndex.from_product([['if', 'gmm', 'svm'], ['Anchor', 'DTOR']]))

for str_model in ['if', 'gmm', 'svm']:
    for str_surrogate in ['Anchor', 'DTOR']:
        for dataset in [main_path.split('/')[-2] for main_path in lst_datasets]:
            for metric in lst_metrics:
                df_table_final_reformat.loc[(dataset, metric), (str_model, str_surrogate)] = \
                    df_table_final.loc[(str_model, metric), (dataset, str_surrogate)]

# Save consolidated result table
df_table_final_reformat.to_csv('/'.join(lst_datasets[0].split('/')[:-2]) + '/result_table' + '.csv')
df_table_final_reformat.to_html('/'.join(lst_datasets[0].split('/')[:-2]) + '/result_table' + '.html')
print(df_table_final_reformat)
