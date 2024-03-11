# Copyright 2024 Intesa Sanpaolo S.p.A
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

import warnings
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import time

class DTOR:
    """
    Decision Tree Outlier Regressor (DTOR) class for explaining instances in anomaly detection models.
    """
    def __init__(self, X_train, y_pred_train, clf_ad, qt=None, lst_categorical=None):
        """
        Initializes the DTOR class with the dataset, predictions, anomaly detector classifier, scaler, and list of categorical variables.

        Args:
            X_train (pd.DataFrame): Training dataset in its original form.
            y_pred_train (pd.Series or np.array): Predictions (0 for normal, 1 for outlier) for the training dataset.
            clf_ad: Anomaly detector classifier.
            qt: Scaler for the dataset.
            lst_categorical (list): List of the names of categorical variables.
        """
        self.X_train = X_train.reset_index(drop=True)
        self.y_pred_train = y_pred_train.reset_index(drop=True) if isinstance(y_pred_train, pd.Series) else y_pred_train
        self.qt = qt
        self.X_train_norm = X_train if qt is None else qt.transform(X_train)
        self.clf_ad = clf_ad
        self.lst_categorical = lst_categorical
        self.global_dts = None

    def explain_instances(self, X_expl, y_pred_expl, beta=1, max_depth=20, splitter='random',
                          min_impurity_decrease=0.001, random_state=5, bool_add_neigh=False, k=3,
                          bool_opposite_neigh=False, bool_internal=True):
        """
        Explains the instances in X_expl.

        Args:
            X_expl (pd.DataFrame): Instances to be explained.
            y_pred_expl (int): Label of the instances (0 for normal, 1 for outlier).
            beta (float): Weight parameter in the fit for the instance to be explained. Default is 1.
            max_depth (int): Maximum depth allowed for the decision tree.
            splitter (str): Type of splitter for the decision tree ('random' or 'best').
            min_impurity_decrease (float): Minimum impurity for a split in the decision tree.
            random_state (int): Random state of the decision tree.
            bool_add_neigh (bool): If True, adds local dataset neighbors of X_expl with the same label.
            k (int): Number of neighbors to search (only those with the same label are added).
            bool_opposite_neigh (bool): If True, adds the nearest neighbors of the opposite label.
            bool_internal (bool): Internal parameter.

        Returns:
            tuple: Tuple containing the rule, decision tree, target of decision tree, and dataset of training of decision tree.
        """
        X_train_dt = self.X_train.copy()
        M = len(X_expl)

        if bool(self.qt):
            if any("cat" in cs for cs in self.qt.named_steps.keys()):
                cat_steps = [cs for cs in self.qt.named_steps.keys() if "cat" in cs]
                if len(cat_steps) > 1:
                    raise ValueError("There should be at most one Categorical step in Transformer")
                X_train_dt = self.qt[cat_steps[0]].transform(X_train_dt)
                X_expl = self.qt[cat_steps[0]].transform(X_expl)

        def fn_aux(j):
            dtj = DecisionTreeRegressor(max_depth=max_depth, criterion='squared_error', splitter=splitter,
                                        max_features=None, min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0, max_leaf_nodes=None,
                                        min_impurity_decrease=min_impurity_decrease, ccp_alpha=0.0,
                                        random_state=random_state)
            sample_weight = [1] * len(X_train_dt) + [(len(X_train_dt) + 1) * beta]
            yj = np.append(self.y_pred_train, y_pred_expl[j])
            Xj = pd.concat([X_train_dt, X_expl.iloc[j:j+1]]).reset_index(drop=True)
            dtj.fit(Xj, yj, sample_weight)
            rule = extract_path(dtj, X_train_dt.columns, X_expl.iloc[j])

            return rule, dtj, Xj, yj

        out = Parallel(n_jobs=-1)(delayed(fn_aux)(j) for j in range(M))

        return tuple(zip(*out))

    def explain_instances_with_precision(self, X_expl, y_pred_expl=1, N_gen=20, random_state=5,
                                         sigma_thresh=0.05, th=0.5, **kwargs):
        """
        Explains the instances with precision.

        Args:
            X_expl (pd.DataFrame): Instances to be explained.
            y_pred_expl (int): Prediction reference of X_expl (0 for normal, 1 for outlier).
            N_gen (int): Number of synthetic dataset generations for precision estimation.
            random_state (int): Random state.
            sigma_thresh (float): Threshold on standard deviation representing a "valid" rule.
            th (float): Threshold for classification.
            kwargs: Additional parameters.

        Returns:
            dict: Dictionary containing various metrics and information about the explanations.
        """
        start = time.time()

        if self.qt is None:
            X_expl_norm = X_expl
        else:
            X_expl_norm = self.qt.transform(X_expl)
        rules, dts, Xs, ys = self.explain_instances(X_expl, y_pred_expl, random_state=random_state, **kwargs)
        lst_categorical = self.lst_categorical if self.lst_categorical else []

        M = len(X_expl)
        out = dict()

        for j in range(M):
            features_of_rule_j = [i.split(' ')[0].replace("`", "") for i in rules[j]]
            dct_grid = {}
            index_conditional = []
            X_aux = Xs[j].copy()

            for k in range(len(rules[j])):
                X_cond = X_aux.astype('float32').query(rules[j][k])
                index_conditional = index_conditional + list(X_cond.index)

                X_cond_feat = X_cond[features_of_rule_j[k]]

                if features_of_rule_j[k] in lst_categorical or \
                        any(f"{feat_name}_" in features_of_rule_j[k] for feat_name in lst_categorical):
                    if features_of_rule_j[k] not in dct_grid.keys():
                        dct_grid[features_of_rule_j[k]] = X_cond_feat.unique()
                    else:
                        X_cond_feat_filtered = X_cond_feat[(X_cond_feat >= min(dct_grid[features_of_rule_j[k]])) &
                                                           (X_cond_feat <= max(dct_grid[features_of_rule_j[k]]))]
                        dct_grid[features_of_rule_j[k]] = np.unique(
                            np.append(X_cond_feat_filtered.values, dct_grid[features_of_rule_j[k]])
                        )
                else:
                    if features_of_rule_j[k] not in dct_grid.keys():
                        dct_grid[features_of_rule_j[k]] = np.unique(
                            np.append(np.quantile(X_cond_feat.values, [0, 0.25, 0.5, 0.75, 1]),
                                      X_cond_feat.mean())
                        )
                    else:
                        X_cond_feat_filtered = X_cond_feat[(X_cond_feat >= min(dct_grid[features_of_rule_j[k]])) &
                                                           (X_cond_feat <= max(dct_grid[features_of_rule_j[k]]))]
                        dct_grid[features_of_rule_j[k]] = np.unique(
                            np.append(np.quantile(X_cond_feat_filtered.values, [0, 0.25, 0.5, 0.75, 1]),
                                      X_cond_feat_filtered.mean())
                        )

            np.random.seed(random_state)
            index_conditional = pd.Series(index_conditional).value_counts()
            len_i_c = len(index_conditional)

            if len_i_c < N_gen:
                index_conditional = list(index_conditional.index) + list(np.random.choice(X_aux.index, N_gen-len_i_c))
            else:
                index_conditional = index_conditional.index[:N_gen]

            dtf_synth = X_aux.copy().loc[index_conditional]
            for k in features_of_rule_j:
                dtf_synth.loc[:, k] = np.random.choice(dct_grid[k], N_gen)

            if dtf_synth.query(" and ".join(rules[j])).shape[0] != dtf_synth.shape[0]:
                print('Error.')

            precision_1 = np.mean(self.clf_ad.score_samples(dtf_synth) < th)
            precision_0 = np.mean(self.clf_ad.score_samples(dtf_synth) >= th)
            pred_mean = np.mean(self.clf_ad.score_samples(dtf_synth))
            pred_std = np.std(self.clf_ad.score_samples(dtf_synth))
            coverage = Xs[j].astype('float32').query(" and ".join(rules[j])).shape[0] / Xs[j].shape[0]
            ad_score = self.clf_ad.score_samples(X_expl_norm.iloc[j:j+1])[0]
            ad_pred_original = self.clf_ad.predict(X_expl_norm.iloc[j:j+1])[0]
            ad_pred = -2*(ad_score < th) + 1

            end = time.time()
            rules_tmp = rules[j]
            rule_not_fulfilled = []

            for rule_tmp in rules_tmp:
                if X_expl.iloc[j:j+1].astype("float64").query(rule_tmp).shape[0] == 0:
                    rule_not_fulfilled.append(rule_tmp)

            out[X_expl.index[j]] = {
                'score_AD': ad_score,
                'score_input': y_pred_expl[j],
                'prediction_hard_AD': ad_pred_original,
                'prediction_hard_AD_check': ad_pred,
                'prediction_surrogate': dts[j].predict(Xs[j].tail(1))[0],
                'prediction_hard_surrogate': dts[j].predict(Xs[j].tail(1))[0] < th,
                'precision_1': precision_1,
                'precision_0': precision_0,
                'precision': precision_1*(ad_pred == -1) + precision_0*(ad_pred == 1),
                'mean_score_rule': pred_mean,
                'std_score_rule': pred_std,
                'z-score_rule': 0 if pred_mean == ad_score else (ad_score - pred_mean) / pred_std,
                'coverage': coverage,
                'sigma_above_thresh': f"{pred_std:.4f} > {sigma_thresh * abs(ad_score):.4f}" if
                pred_std > (sigma_thresh * abs(ad_score)) else '',
                'rule_length': len(rules[j]),
                'rule': rules[j],
                'Rules not fulfilled': ' AND '.join(rule_not_fulfilled),
                'Execution_time': end - start
            }

        return out

    def explain_instances_multiple_seed(self, X_expl, y_pred_expl=1, N_gen=20, seeds=5, seed=42,
                                        sigma_thresh=0.05, **kwargs):
        """
        Extracts multiple explanations with several seeds and picks the "best" one.

        Args:
            X_expl (pd.DataFrame): Instances to be explained.
            y_pred_expl (int): Prediction reference of X_expl (0 for normal, 1 for outlier).
            N_gen (int): Number of synthetic dataset generations for precision estimation.
            seeds (int, list): The number of seeds for which explanations are extracted, or the explicit list of seeds.
            seed (int): If seeds is an integer, this seed is the random seed that governs the extraction of the seeds.
            sigma_thresh (float): Threshold on standard deviation representing a "valid" rule.
            kwargs: Additional parameters.

        Returns:
            dict: Dictionary with keys 'Score_AD', 'Prediction_AD', 'Prediction_dtol', 'mean_score_rule',
            'std_score_rule', 'z-score_rule', 'coverage', 'sigma_above_thresh', 'DTOL_rule'.
        """
        np.random.seed(seed)

        if isinstance(seeds, int):
            seeds = np.random.randint(max(100000, seeds), size=seeds)
        elif not isinstance(seeds, list):
            raise ValueError("seeds should be a list of integers or an integer")

        out = {k: {} for k in X_expl.index}
        zs = {k: [] for k in X_expl.index}
        zs_above_thresh = {k: [] for k in X_expl.index}

        for s in seeds:
            out_s = self.explain_instances_with_precision(X_expl, y_pred_expl=y_pred_expl, N_gen=N_gen,
                                                          random_state=s, sigma_thresh=sigma_thresh, **kwargs)
            for idx, out_idx in out_s.items():
                out[idx][s] = out_idx
                if not bool(out_idx["sigma_above_thresh"]):
                    zs[idx].append((s, abs(out_idx["z-score_rule"]), len(out_idx["DTOL_rule"])))
                else:
                    zs_above_thresh[idx].append((s, abs(out_idx["z-score_rule"]), out_idx["rule_length"]))

        res = dict()
        for idx, out_idx_s in out.items():
            if bool(zs[idx]):
                zs_ord = sorted(zs[idx], key=lambda x: (x[1], x[2]))
                res[idx] = out_idx_s[zs_ord[0][0]]
            else:
                print(f"No rule satisfying the sigma threshold has been found for index {idx}, "
                      "return the rule with closest prediction anyway")
                zs_above_thresh_ord = sorted(zs_above_thresh[idx], key=lambda x: (x[1], x[2]))
                res[idx] = out_idx_s[zs_above_thresh_ord[0][0]]

        return res

    def global_surrogate_predict(self, X):
        """
        Predicts using the global surrogate model.

        Args:
            X (pd.DataFrame): Input instances.

        Returns:
            np.array: Predicted values.
        """
        if self.global_dts is None:
            warnings.warn("No global surrogate trees available: global_surrogate_fit method is called with default"
                          "parameters")
            self.global_surrogate_fit()

        if bool(self.qt):
            if any("cat" in cs for cs in self.qt.named_steps.keys()):
                cat_steps = [cs for cs in self.qt.named_steps.keys() if "cat" in cs]
                if len(cat_steps) > 1:
                    raise ValueError("There should be at most one Categorical step in Transformer")
                X_dt = self.qt[cat_steps[0]].transform(X)
        else:
            X_dt = X

        len_dt = len(self.global_dts)
        len_X = X.shape[0]
        mat_output = np.zeros((len_X, len_dt))

        for c, dt in enumerate(self.global_dts):
            mat_output[:, c] = dt.predict(X_dt)

        output_regr = np.quantile(mat_output, 0.05, axis=1)
        return output_regr

    def global_surrogate_fit(self, outliers_num=0.1, seed=42):
        """
        Builds a global surrogate model using Decision Tree Outlier Regressor (DTOR) as an ensemble.
    
        Parameters:
        - outliers_num: Percentage of outliers in the training dataset (default=0.1).
        - seed: Random seed for reproducibility (default=42).
    
        Returns:
        - None
        """
        if isinstance(outliers_num, int):
            n_outlier = min(outliers_num, int(np.floor(self.X_train.shape[0] / 2)))
        elif outliers_num < 1:
            n_outlier = int(np.ceil(outliers_num * self.X_train.shape[0]))
            n_outlier = min(n_outlier, int(np.floor(self.X_train.shape[0] / 2)))
        idx_outliers = np.argsort(self.y_pred_train)[:n_outlier]
        np.random.seed(seed)
        idx_random = np.random.choice(range(self.X_train.shape[0]), n_outlier)
        idx = np.concatenate([idx_outliers, idx_random])
        xtrain_original_outliers = self.X_train.copy().iloc[idx, :].reset_index(drop=True)
    
        dtol_config = {'max_depth': 12, 'splitter': 'random', 'min_impurity_decrease': 0.00, 'beta': 0.1}
        _, dts, _, _ = self.explain_instances(xtrain_original_outliers, y_pred_expl=self.y_pred_train[idx], **dtol_config)
        self.global_dts = dts

