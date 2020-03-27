import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn import model_selection, metrics, svm, datasets, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import shap
import graphviz
from sklearn.tree import export_graphviz

method_flag = 7  # 0: LightGBM, 1: XGBoost, 2: scikit-learn 3:LDA, 4: linear SVM, 5: nonlinear SVM
                 # 6: DecisionTree 7: RandomForest 8: ExtraTreesClassifier 9: KNeighborsClassifier
optimization_method = 2 # 1: GridSearchCV, 2: Optuna
trials = 25
fold_number = 5  # "fold_number"-fold cross-validation
number_of_sub_models = 100
number_of_test_samples = 0.2
fraction_of_validation_samples = 0.2
linear_svm_cs = 2 ** np.arange(-5, 5, dtype=float)  # C for linear svr
nonlinear_svm_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svm_gammas = 2 ** np.arange(-20, 10, dtype=float)  # gamma for nonlinear svr
dt_max_max_depth = 30  # 木の深さの最大値、の最大値
dt_min_samples_leaf = 3  # 葉ごとのサンプル数の最小値
max_number_of_k = 20  # 使用する k の最大値

# load iris dataset
iris = datasets.load_iris()
x = iris.data
x = pd.DataFrame(x)
x.columns = iris.feature_names
y = iris.target

# Divide samples into training samples and test samples
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=number_of_test_samples, random_state=0)

#autoscaling
autocalculated_train_x = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0, ddof=1)
autocalculated_test_x = (test_x - train_x.mean(axis=0)) / train_x.std(axis=0, ddof=1)

# hyperparameter optimization with optuna and modeling
if method_flag == 0:  # LightGBM
    train_x_tmp, train_x_validation, train_y_tmp, train_y_validation = train_test_split(train_x,
                                                                                        train_y,
                                                                                        test_size=fraction_of_validation_samples,
                                                                                        random_state=0)
    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = lgb.LGBMClassifier(n_estimators = 1000)
        model.fit(train_x_tmp, train_y_tmp, eval_set=(train_x_validation, train_y_validation), eval_metric = 'logloss', early_stopping_rounds = 100)
        best_n_estimators_in_cv = model.best_iteration_

    def objective(trial):
        param = {
            #'objective': 'multiclass',
            #'metric': 'multi_logloss',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
            #'num_boost_round': trial.suggest_int('num_boost_round', 10, 100000),
            #'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100000),
            #'min_child_samples': trial.suggest_int('min_child_samples', 5, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 500),
        }

        if param['boosting_type'] == 'dart':
            param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if param['boosting_type'] == 'goss':
            param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

        model = lgb.LGBMClassifier(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(model, train_x, train_y, cv=fold_number)
        accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
        return 1.0 - accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = lgb.LGBMClassifier(n_estimators = 1000)
        model.fit(train_x_tmp, train_y_tmp, eval_set=(train_x_validation, train_y_validation), eval_metric = 'multi_logloss', early_stopping_rounds = 100)
        best_n_estimators_in_cv = model.best_iteration_

    model = lgb.LGBMClassifier(**study.best_params, n_estimators = best_n_estimators_in_cv)

elif method_flag == 1:  # XGBoost
    train_x_tmp, train_x_validation, train_y_tmp, train_y_validation = train_test_split(train_x,
                                                                                        train_y,
                                                                                        test_size=fraction_of_validation_samples,
                                                                                        random_state=0)
    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = xgb.XGBClassifier(n_estimators=1000)
        model.fit(train_x_tmp, train_y_tmp,
                  eval_set=[(train_x_validation, train_y_validation.reshape([len(train_y_validation), 1]))],
                  eval_metric='mlogloss', early_stopping_rounds=100)
        best_n_estimators_in_cv = model.best_iteration
    def objective(trial):
        param = {
            'silent': 1,
            #            'objective': 'binary:logistic',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
        }

        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

        model = xgb.XGBClassifier(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(model, train_x, train_y, cv=fold_number)
        accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
        return 1.0 - accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    if fraction_of_validation_samples == 0:
        best_n_estimators = number_of_sub_models
    else:
        model = xgb.XGBClassifier(**study.best_params, n_estimators=1000)
        model.fit(train_x_tmp, train_y_tmp,
                  eval_set=[(train_x_validation, train_y_validation.reshape([len(train_y_validation), 1]))],
                  eval_metric='mlogloss', early_stopping_rounds=100)
        best_n_estimators = model.best_iteration

    model = xgb.XGBClassifier(**study.best_params, n_estimators = best_n_estimators)

elif method_flag == 2:  # scikit-learn

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = GradientBoostingClassifier(n_estimators=1000, validation_fraction=fraction_of_validation_samples,
                                          n_iter_no_change=100)
        model.fit(train_x, train_y)
        best_n_estimators_in_cv = len(model.estimators_)

    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 20),
            'max_features': trial.suggest_loguniform('max_features', 0.1, 1.0)
        }

        model = GradientBoostingClassifier(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(model, train_x, train_y, cv=fold_number)
        accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
        return 1.0 - accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    if fraction_of_validation_samples == 0:
        best_n_estimators_in_cv = number_of_sub_models
    else:
        model = GradientBoostingClassifier(n_estimators=1000, validation_fraction=fraction_of_validation_samples,
                                          n_iter_no_change=100)
        model.fit(train_x, train_y)
        best_n_estimators_in_cv = len(model.estimators_)

    model = GradientBoostingClassifier(**study.best_params, n_estimators = best_n_estimators_in_cv)

elif method_flag == 3:  # LDA
    model = LinearDiscriminantAnalysis()

elif method_flag == 4:  # linear SVM
    linear_svm_in_cv = GridSearchCV(svm.SVC(kernel='linear'), {'C': linear_svm_cs}, cv=fold_number)
    linear_svm_in_cv.fit(autocalculated_train_x, train_y)
    optimal_linear_svm_c = linear_svm_in_cv.best_params_['C']
    model = svm.SVC(kernel='linear', C=optimal_linear_svm_c)

elif method_flag == 5:  # nonlinear SVM
    if optimization_method == 1:
        nonlinear_svm_in_cv = GridSearchCV(svm.SVC(kernel='rbf'), {'C': nonlinear_svm_cs, 'gamma': nonlinear_svm_gammas},
                                       cv=fold_number)
        nonlinear_svm_in_cv.fit(autocalculated_train_x, train_y)
        optimal_nonlinear_svm_c = nonlinear_svm_in_cv.best_params_['C']
        optimal_nonlinear_svm_gamma = nonlinear_svm_in_cv.best_params_['gamma']
        model = svm.SVC(kernel='rbf', C=optimal_nonlinear_svm_c, gamma=optimal_nonlinear_svm_gamma)

    elif optimization_method == 2:
        #optuna
        def objective(trial):
            param = {
                    'kernel': 'rbf',
                    'C': trial.suggest_loguniform('C', 1e-2, 1e2),
                    'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e1),
            }

            model = svm.SVC(**param)
            estimated_y_in_cv = model_selection.cross_val_predict(model, autocalculated_train_x, train_y, cv=fold_number)
            accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
            return 1.0 - accuracy

        study = optuna.create_study()
        study.optimize(objective, n_trials=trials)

        model = svm.SVC(kernel='rbf', **study.best_params)



elif method_flag == 6:  # Decision tree
    # クロスバリデーションによる木の深さの最適化
    accuracy_all = []
    for max_depth in range(1, dt_max_max_depth):
        model_in_cv = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=dt_min_samples_leaf)
        estimated_y_in_cv = model_selection.cross_val_predict(model_in_cv, train_x, train_y,
                                                              cv=fold_number)
        accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
        accuracy_all.append(accuracy)
        #r2cv_all.append(1 - sum((train_y - estimated_y_in_cv) ** 2) / sum((train_y - train_y.mean()) ** 2))
    optimal_max_depth = np.where(accuracy_all == np.max(accuracy_all))[0][0] + 2  # r2cvが最も大きい木の深さ
    model = tree.DecisionTreeClassifier(max_depth=optimal_max_depth,
                                                  min_samples_leaf=dt_min_samples_leaf)  # DTモデルの宣言

elif method_flag == 7:  # RandomForest
    if optimization_method == 1:
        param = {'max_depth': [2,3, None],
                 'n_estimators': [20, 50, 100, 200, 500, 1000],
                 #'max_features': [1, 3, 10, 20],
                 'min_samples_split' : [2, 3, 10],
                 'min_samples_leaf' : [1, 3, 10],
                 'bootstrap' : [True, False],
                 'criterion': ['gini', 'entropy']}

        RFC_in_cv = GridSearchCV(RandomForestClassifier(random_state=0), param_grid = param,
                                 scoring = 'accuracy', cv = fold_number, n_jobs = 1)
        RFC_in_cv.fit(train_x, train_y)
        model = RandomForestClassifier(**RFC_in_cv.best_params_)

    elif optimization_method == 2:
        #optuna
        def objective(trial):
            param = {
                    'n_estimators': trial.suggest_int('n_estimators', 20, 1000),
                    'max_features': trial.suggest_uniform('max_features', 0.1, 0.9),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'max_depth': trial.suggest_int('max_depth', 1, 5),
            }

            model = RandomForestClassifier(**param)
            estimated_y_in_cv = model_selection.cross_val_predict(model, train_x, train_y, cv=fold_number)
            accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
            return 1.0 - accuracy

        study = optuna.create_study()
        study.optimize(objective, n_trials=trials)

        model = RandomForestClassifier(**study.best_params)

elif method_flag == 8:  # Extremely Randomized Tree
    #optuna
    def objective(trial):
        param = {
                'n_estimators': trial.suggest_int('n_estimators', 20, 400),
                'max_features': trial.suggest_uniform('max_features', 0.1, 0.9),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 50),
            }

        model = ExtraTreesClassifier(**param)
        estimated_y_in_cv = model_selection.cross_val_predict(model, train_x, train_y, cv=fold_number)
        accuracy = metrics.accuracy_score(train_y, estimated_y_in_cv)
        return 1.0 - accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=trials)

    model = ExtraTreesClassifier(**study.best_params)

elif method_flag == 9:  # KNeighbors
    # CV による k の最適化
    accuracy_in_cv_all = []  # 空の list の変数を作成して、成分数ごとのクロスバリデーション後の 正解率 をこの変数に追加していきます
    ks = []  # 同じく k の値をこの変数に追加していきます
    for k in range(1, max_number_of_k + 1):
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')  # k-NN モデルの宣言
        # クロスバリデーション推定値の計算し、DataFrame型に変換
        estimated_y_in_cv = pd.DataFrame(model_selection.cross_val_predict(model, autocalculated_train_x, train_y,
                                                                       cv=fold_number))
        accuracy_in_cv = metrics.accuracy_score(train_y, estimated_y_in_cv)  # 正解率を計算
        print(k, accuracy_in_cv)  # k の値と r2 を表示
        accuracy_in_cv_all.append(accuracy_in_cv)  # r2 を追加
        ks.append(k)  # k の値を追加

    def plot_and_selection_of_hyperparameter(hyperparameter_values, metrics_values, x_label, y_label):
        # ハイパーパラメータ (成分数、k-NN の k など) の値ごとの統計量 (CV 後のr2, 正解率など) をプロット
        plt.rcParams['font.size'] = 18
        plt.scatter(hyperparameter_values, metrics_values, c='blue')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        # 統計量 (CV 後のr2, 正解率など) が最大のときのハイパーパラメータ (成分数、k-NN の k など) の値を選択
        return hyperparameter_values[metrics_values.index(max(metrics_values))]

    # k の値ごとの CV 後の正解率をプロットし、CV 後の正解率が最大のときを k の最適値に
    optimal_k = plot_and_selection_of_hyperparameter(ks, accuracy_in_cv_all, 'k',
                                                                  'cross-validated accuracy')
    print('\nCV で最適化された k :', optimal_k, '\n')

    # k-NN
    model = KNeighborsClassifier(n_neighbors=optimal_k, metric='euclidean')  # モデルの宣言

if method_flag <= 2:
    clf = model.fit(train_x, train_y)
    calculated_y_train = clf.predict(train_x)
    predicted_y_test = clf.predict(test_x)
elif method_flag > 2:
    clf = model.fit(autocalculated_train_x, train_y)
    calculated_y_train = clf.predict(autocalculated_train_x)
    predicted_y_test = clf.predict(autocalculated_test_x)

# confusion matrix for training data
confusion_matrix_train = metrics.confusion_matrix(train_y, calculated_y_train, labels=sorted(set(train_y)))
print('training samples')
print(sorted(set(train_y)))
print(confusion_matrix_train)

# estimated_y in cross-validation
if method_flag <= 2:
    estimated_y_in_cv = model_selection.cross_val_predict(clf, train_x, train_y, cv=fold_number)
if method_flag > 2:
    estimated_y_in_cv = model_selection.cross_val_predict(clf, autocalculated_train_x, train_y, cv=fold_number)
# confusion matrix in cross-validation
confusion_matrix_train_in_cv = metrics.confusion_matrix(train_y, estimated_y_in_cv, labels=sorted(set(train_y)))
print('training samples in CV')
print(sorted(set(train_y)))
print(confusion_matrix_train_in_cv)

# confusion matrix for test data
confusion_matrix_test = metrics.confusion_matrix(test_y, predicted_y_test, labels=sorted(set(train_y)))
print('')
print('test samples')
print(sorted(set(train_y)))
print(confusion_matrix_test)

#visualization
if method_flag == 0:
    lgb.plot_importance(clf, figsize = (18,8), max_num_features=30)
    plt.show()

if method_flag == 1:
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    xgb.plot_importance(clf, ax = ax, max_num_features = 30, importance_type = 'weight')
    plt.show()

if method_flag == 2:
    feature_importances = clf.feature_importances_
    # make importances relative to max importance
    feature_importances = 100.0 * (feature_importances / feature_importances.max())
    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importances[sorted_idx], align='center')
    #plt.yticks(pos, iris.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

if method_flag == 6:
    """
    #Tree Plot
    fig = plt.figure(figsize=figure.figaspect(1))
    ax = fig.add_subplot()
    plot_tree(clf, feature_names=boston.feature_names, ax=ax, filled=True);
    plt.show()
    """

    #Graphviz
    dot_data = export_graphviz(
                        clf,
                        class_names=iris.target_names,
                        #feature_names=boston.feature_names,
                        filled=True,
                        rounded=True,
                        out_file=None
                    )
    graph = graphviz.Source(dot_data)
    graph.render("iris-tree", format="png")

if method_flag == 7:
    # 説明変数の重要度
    x_importances = pd.DataFrame(clf.feature_importances_, index=pd.DataFrame(train_x).columns, columns=['importance'])
    x_importances.to_csv('rf_x_importances.csv')  # csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください

#SHAP visualization
if method_flag == 3 or method_flag == 4 or method_flag == 5 or method_flag == 9:
    explainer = shap.KernelExplainer(clf.predict, train_x)
    shap_values = explainer.shap_values(train_x.loc[[0]])
    shap.force_plot(explainer.expected_value, shap_values[0], train_x.loc[[0]], matplotlib = True)

"""
    shap_values = explainer.shap_values(train_x)
    shap.summary_plot(shap_values, features = train_x,
                #plot_type = 'bar'
                )

    shap.dependence_plot(ind="RM", shap_values=shap_values, features = train_x,
                    interaction_index = 'TSTAT',
                    )
"""

if 0 <= method_flag <= 1 or 6 <= method_flag <= 8:
    explainer = shap.TreeExplainer(clf)

    #shap_values = explainer.shap_values(train_x.loc[[0]])
    #shap.force_plot(explainer.expected_value, shap_values[0], train_x.loc[[0]], matplotlib = True)

    shap_values = explainer.shap_values(train_x)
    shap.summary_plot(shap_values, features = train_x,
                    plot_type = 'bar'
                    )

"""
    shap.dependence_plot(ind="petal_width_", shap_values=shap_values, features = train_x,
                    interaction_index = 'sepal_width_',
                    )

    shap.decision_plot(explainer.expected_value, shap_values[1][0:20], train_x[0:20],
                       #link="logit",
                       #highlight=misclassified[0:20]
                    )
"""




"""
#submit file
submit_file = pd.DataFrame({'Id' : test_id , 'revenue' : predicted_y_test})
submit_file.to_csv('Officesci.csv', index = False)
"""
