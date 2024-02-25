import warnings
import numpy as np
import pandas as pd
import math
import os
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

matplotlib.use('agg')

def extract_path(tree, feature_names, instance):
    path = []
    node_id = 0  # Start at the root
    while True:
        if tree.tree_.children_left[node_id] == -1:  # Leaf node
            break

        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]
        
        # Round the threshold
        delta = abs(instance[feature] - threshold)
        order_of_mag = math.floor(math.log10(delta))
        multiplier = 10 ** order_of_mag if order_of_mag >= 0 else 10 ** (-order_of_mag)
        threshold_rounded = round(threshold * multiplier) / multiplier
        
        # Add condition to path
        condition = f"`{feature_names[feature]}` {'<=' if instance[feature] <= threshold else '>'} {threshold_rounded}"
        path.append(condition)

        # Move to next node
        node_id = tree.tree_.children_left[node_id] if instance[feature] <= threshold else tree.tree_.children_right[node_id]

    return path


class DTOLR:
    """
    Build a decision tree that separates the instance to be explained from the other instances with a different label.
    """

    def __init__(self, X_train, y_pred_train, clf_ad, qt=None, lst_categorical=None):
        """
        Init the class with the dataset, prediction, classified, transformed and list of categorical variables
        """

        self.X_train = X_train.reset_index(drop=True)
        self.y_pred_train = y_pred_train.reset_index(drop=True) if isinstance(y_pred_train, pd.Series) else y_pred_train
        self.qt = qt
        self.X_train_norm = qt.transform(self.X_train) if qt else X_train
        self.clf_ad = clf_ad
        self.lst_categorical = lst_categorical
        self.global_dts = None

    def explain_instances(self, X_expl, y_pred_expl, beta=1, max_depth=20, splitter='random',
                          min_impurity_decrease=0.001,
                          random_state=5, bool_add_neigh=False,
                          k=3,
                          bool_opposite_neigh=False,
                          bool_internal=True):
        """
        Explain the instance X_expl. A Decision Tree is trained per each instance.
        """

        # attach the instance to explain as the last row of the train set
        X_train_dt = self.X_train.copy()

        # number of instances to explain
        M = len(X_expl)

        if bool(self.qt) and any("cat" in cs for cs in self.qt.named_steps.keys()):
            cat_steps = [cs for cs in self.qt.named_steps.keys() if "cat" in cs]
            if len(cat_steps) > 1:
                raise ValueError("There should be at most one Categorical step in Transformer")
            X_train_dt = self.qt[cat_steps[0]].transform(X_train_dt)
            X_expl = self.qt[cat_steps[0]].transform(X_expl)

        len_dataset = X_train_dt.shape[0]
        if len_dataset < 1:
            print("No rows in the dataset.")
            raise ValueError("No rows in the dataset.")

        if beta <= 0:
            print('Warning, beta should be > 0. Set to 1.')
            beta = 1

        def fn_aux(j):
            dtj = DecisionTreeRegressor(max_depth=max_depth,
                                        criterion='squared_error',
                                        splitter=splitter,
                                        min_impurity_decrease=min_impurity_decrease,
                                        random_state=random_state)
            sample_weight = [1] * len_dataset + [(len_dataset + 1) * beta]
            yj = np.append(self.y_pred_train, y_pred_expl[j])
            Xj = pd.concat([X_train_dt, X_expl.iloc[j:j + 1]]).reset_index(drop=True)
            dtj.fit(Xj, yj, sample_weight)
            rule = extract_path(dtj, X_train_dt.columns, X_expl.iloc[j])

            return rule, dtj, Xj, yj

        out = Parallel(n_jobs=-1)(delayed(fn_aux)(j) for j in range(M))

        return tuple(zip(*out))

    def explain_instances_with_precision(self, X_expl, y_pred_expl=1, N_gen=20,
                                         random_state=5, sigma_thresh=0.05, **kwargs):
        """
        Explain the instance as in explain_instance, but it is added information about precision and coverage.
        Call to the classifier is needed.
        """

        X_expl_norm = self.qt.transform(X_expl) if self.qt else X_expl
        rules, dts, Xs, ys = self.explain_instances(X_expl, y_pred_expl,
                                                    random_state=random_state, **kwargs)

        if self.lst_categorical is not None:
            lst_categorical = self.lst_categorical

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

                if features_of_rule_j[k] in lst_categorical or \
                        any(f"{feat_name}_" in features_of_rule_j[k] for feat_name in lst_categorical):
                    dct_grid[features_of_rule_j[k]] = X_cond.unique()
                else:
                    dct_grid[features_of_rule_j[k]] = np.unique(
                        np.append(np.quantile(X_cond.values, [0, 0.25, 0.5, 0.75, 1]),
                                  X_cond.mean())
                    )

            np.random.seed(random_state)
            index_conditional = pd.Series(index_conditional).value_counts()

            len_i_c = len(index_conditional)
            if len_i_c < N_gen:
                index_conditional = list(index_conditional.index) + list(np.random.choice(X_aux.index, N_gen - len_i_c))
            else:
                index_conditional = index_conditional.index[:N_gen]

            dtf_synth = X_aux.copy().loc[index_conditional]
            for k in features_of_rule_j:
                dtf_synth.loc[:, k] = np.random.choice(dct_grid[k], N_gen)

            if bool(self.qt) and any("cat" in cs for cs in self.qt.named_steps.keys()):
                num_steps = [ns for ns in self.qt.named_steps.keys() if "cat" not in ns]
                if len(num_steps) > 1:
                    raise ValueError("There should be at most one non-Categorical step in Transformer")
                dtf_synth_norm = self.qt[num_steps[0]].transform(dtf_synth)
            else:
                dtf_synth_norm = self.qt.transform(dtf_synth) if self.qt else dtf_synth
                if not isinstance(dtf_synth_norm, pd.DataFrame):
                    dtf_synth_norm = pd.DataFrame(self.qt.transform(dtf_synth), columns=dtf_synth.columns)

            pred_mean = np.mean(self.clf_ad.score_samples(dtf_synth_norm))
            pred_std = np.std(self.clf_ad.score_samples(dtf_synth_norm))
            coverage = Xs[j].astype('float32').query(" and ".join(rules[j])).all(axis=1).mean()
            ad_score = self.clf_ad.score_samples(X_expl_norm.iloc[j:j + 1])[0]
            ad_pred = self.clf_ad.predict(X_expl_norm.iloc[j:j + 1])[0]

            out[X_expl.index[j]] = {
                'Score_AD': ad_score,
                'Score_input': y_pred_expl[j],
                'Prediction_AD': ad_pred,
                'Prediction_dtol': dts[j].predict(Xs[j].tail(1))[0],
                'mean_score_rule': pred_mean,
                'std_score_rule': pred_std,
                'z-score_rule': 0 if pred_mean == ad_score else (ad_score - pred_mean) / pred_std,
                'coverage': coverage,
                'sigma_above_thresh': f"{pred_std:.4f} > {sigma_thresh * abs(ad_score):.4f}" if
                pred_std > (sigma_thresh * abs(ad_score)) else '',
                'rule_length': len(rules[j]),
                'DTOL_rule': rules[j]
            }

        return out

    def explain_instances_multiple_seed(self, X_expl, y_pred_expl=1, N_gen=20, seeds=5, seed=42,
                                        sigma_thresh=0.05, **kwargs):
        """Extracts multiple explanations with several seeds and pick the "best" one"""

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
        if self.global_dts is None:
            warnings.warn("No global surrogate trees available: global_surrogate_fit method is called with default"
                          "parameters")
            self.global_surrogate_fit()

        if bool(self.qt) and any("cat" in cs for cs in self.qt.named_steps.keys()):
            cat_steps = [cs for cs in self.qt.named_steps.keys() if "cat" in cs]
            if len(cat_steps) > 1:
                raise ValueError("There should be at most one Categorical step in Transformer")
            X_dt = self.qt[cat_steps[0]].transform(X)

        len_dt = len(self.global_dts)
        len_X = X.shape[0]
        mat_output = np.zeros((len_X, len_dt))

        for c, dt in enumerate(self.global_dts):
            mat_output[:, c] = dt.predict(X_dt)

        output_regr = np.quantile(mat_output, 0.05, axis=1)

        return output_regr

    def global_surrogate_fit(self, outliers_num=0.1, seed=42):
        """Builds a surrogate model using several DTLOR as ensemble"""

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
        _, dts, _, _ = self.explain_instances(xtrain_original_outliers,
                                              y_pred_expl=self.y_pred_train[idx],
                                              **dtol_config)
        self.global_dts = dts


if __name__ == '__main__':
    # Load dataset
    # X_train, X_test = ...
    clf_ad = IsolationForest(random_state=0, n_jobs=-1, contamination=0.05)
    clf_ad.fit(X_train)
    # save the prediction
    y_train = clf_ad.predict(X_train)
    y_test = clf_ad.predict(X_test)
    # IF output is anomaly -1 vs normal 1, let's change it to 1 vs 0
    y_train = 1 * (y_train == -1)
    y_test = 1 * (y_test == -1)
    y_train_score = clf_ad.score_samples(X_train)
    y_test_score = clf_ad.score_samples(X_test)

    dtol = DTOLR(X_train_original, y_train_score, clf_ad, qt=preprocessor, lst_categorical=lst_categorical)
    idx_expl = range(10)
    X_expl = X_test.iloc[idx_expl]
    X_expl_original = X_test_original.iloc[idx_expl]
    y_pred_expl = clf_ad.score_samples(X_expl)
    print("label to explain: ", y_pred_expl)
    dtol_config = {'max_depth': 12, 'splitter': 'random', 'min_impurity_decrease': 0.00, 'beta': 0.1}
    # Complete output with precision
    output = dtol.explain_instances_with_precision(X_expl_original, y_pred_expl,
                                                   N_gen=100,
                                                   random_state=5,
                                                   **dtol_config)

    output_seed = dtol.explain_instances_multiple_seed(X_expl_original, y_pred_expl, N_gen=100, **dtol_config)

    print(f"explanations for IDs {idx_expl}, with random seed")
    print(pd.DataFrame(output).T)

    print(f"explanations for IDs {idx_expl}, optimized over several seeds")
    print(pd.DataFrame(output_seed).T)
