from xgboost import XGBClassifier

def get_baseline_A():
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        eval_metric='logloss'
    )
    model.set_params(early_stopping_rounds=20)
    return model
