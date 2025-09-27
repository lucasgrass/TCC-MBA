# all_onehot_orig
# all_onehot_under
# all_onehot_over
# all_label_category_orig
# all_label_category_under
# all_label_category_over
# all_onehot_scaled_orig
# all_onehot_scaled_under
# all_onehot_scaled_over
# top10_onehot_orig
# top10_onehot_under
# top10_onehot_over
# top10_label_category_orig
# top10_label_category_under
# top10_label_category_over
# top10_onehot_scaled_orig
# top10_onehot_scaled_under
# top10_onehot_scaled_over
# top10_fe_onehot_orig
# top10_fe_onehot_under
# top10_fe_onehot_over
# top10_fe_label_category_orig
# top10_fe_label_category_under
# top10_fe_label_category_over
# top10_fe_onehot_scaled_orig
# top10_fe_onehot_scaled_under
# top10_fe_onehot_scaled_over

experiments = [
    {
        "model": "random_forest",
        "dataset": "all_onehot_scaled_over",
        "parameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "class_weight": {0:1, 1:4}
        },
        "threshold": 0.5
    },
    {
        "model": "xgboost",
        "dataset": "top10_fe_label_category_over",
        "parameters": {
            "n_estimators": 2000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 5,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "enable_categorical": True
        },
        "threshold": 0.6
    }
]

