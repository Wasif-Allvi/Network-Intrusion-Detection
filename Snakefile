rule all:
    input:
        'results/exploratory_data_analysis/data_head.csv',
        'results/exploratory_data_analysis/data_description.csv',
        'results/model_fit_evaluation/random_forest_model.pkl',
        'results/model_fit_evaluation/classification_report.txt',
        'results/model_fit_evaluation/confusion_matrix.csv',
        'results/predict_test_data/random_check.csv',
        'results/predict_test_data/test_predictions.csv'

rule data_exploration:
    output:
        'results/exploratory_data_analysis/data_head.csv',
        'results/exploratory_data_analysis/data_description.csv'
    script:
        'data_exploration.py'

rule model_fit:
    output:
        'results/model_fit_evaluation/random_forest_model.pkl',
        'results/model_fit_evaluation/X_test.pkl',
        'results/model_fit_evaluation/y_test.pkl'
    script:
        'model_fit.py'

rule model_evaluation:
    input:
        'results/model_fit_evaluation/random_forest_model.pkl',
        'results/model_fit_evaluation/X_test.pkl',
        'results/model_fit_evaluation/y_test.pkl'
    output:
        'results/model_fit_evaluation/classification_report.txt',
        'results/model_fit_evaluation/confusion_matrix.csv'
    script:
        'model_evaluation.py'

rule random_check:
    input:
        'results/model_fit_evaluation/random_forest_model.pkl'
    output:
        'results/predict_test_data/random_check.csv'
    script:
        'random_check_one_train_data.py'

rule predict_test:
    input:
        'results/model_fit_evaluation/random_forest_model.pkl',
        'results/model_fit_evaluation/X_test.pkl'
    output:
        'results/predict_test_data/test_predictions.csv'
    script:
        'prediction_test_data.py'
