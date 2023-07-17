import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pydotplus
import logging

# Constants
CONFIG_FILE_PATH = "config.json"
OUTPUT_FOLDER = "pdf_outputs"


def load_data(file_path: str, num_features: int) -> pd.DataFrame:
    # Load the dataset
    balance_data = pd.read_csv(file_path, sep=',', header=0)

    # Drop rows with missing values
    balance_data = balance_data.dropna()

    return balance_data


def encode_targets(targets: list) -> list:
    Y_encoded = []
    for Y in targets:
        unique_values = np.unique(Y)
        num_classes = len(unique_values)
        encoded = np.zeros((Y.shape[0], num_classes))
        for j, val in enumerate(unique_values):
            encoded[Y == val, j] = 1
        Y_encoded.append(encoded)
    return Y_encoded


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    with open(CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)

    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth", 3)
    min_samples_leaf = config.get("min_samples_leaf", 5)
    random_state = config.get("random_state", 100)

    clf_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                    max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    clf_rf.fit(X, y)
    return clf_rf


def get_top_features(clf: RandomForestClassifier, feature_names: list, num_top_features: int) -> pd.Series:
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_columns = importance_df.sort_values(by='Importance', ascending=False).head(num_top_features)['Feature']
    return top_columns


def save_decision_tree(clf: RandomForestClassifier, feature_names: list, unique_values: np.ndarray,
                       output_path: str):
    dot_data = export_graphviz(clf.estimators_[0], out_file=None, feature_names=feature_names,
                               class_names=unique_values, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(output_path)


def save_output_to_pdf(output: str, output_path: str):
    plt.figure(figsize=(8.27, 11.69))
    plt.text(0.5, 0.5, output, fontsize=12, ha='center', va='center', wrap=True, fontfamily='serif')
    plt.axis('off')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.5)


def get_file_path() -> str:
    # Read the file path from the configuration file
    with open(CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)
        return config["file_path"]


def create_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


def configure_logging():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


def get_top_feature_values(clf: RandomForestClassifier, feature_names: list, X_test: np.ndarray,
                           num_values: int) -> dict:
    top_feature_values = {}
    for feature_idx, feature_name in enumerate(feature_names):
        feature_values = X_test[:, feature_idx]
        value_counts = pd.Series(feature_values).value_counts()
        top_values = value_counts.head(num_values)
        top_feature_values[feature_name] = top_values
    return top_feature_values


def main():
    configure_logging()
    create_output_folder()

    # Read the config file
    with open(CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)
        NUM_FEATURES = config.get("num_features")
        Y_COLUMNS = config.get("Y_columns", [])
        SUMMARY_NAME = config.get("summary_name", "Dataset")
        NUM_TOP_FEATURES = config.get("num_top_features", 5)
        NUM_TOP_FEATURE_VALUES = config.get("num_top_feature_values", 5)

    # Load the dataset
    file_path = get_file_path()
    original_data = pd.read_csv(file_path, sep=',', header=0)

    # Original dataset information
    original_num_rows, original_num_columns = original_data.shape

    # Drop rows with missing values
    balance_data = load_data(file_path, NUM_FEATURES)
    num_rows_dropped = original_num_rows - balance_data.shape[0]
    num_rows_after_dropping = balance_data.shape[0]
    num_columns = balance_data.shape[1]

    # Export the new list without dropped rows to a CSV file
    remaining_rows_path = os.path.join(OUTPUT_FOLDER, 'remaining_rows.csv')
    balance_data.to_csv(remaining_rows_path, index=False)

    output = f"{SUMMARY_NAME} Dataset Summary:\n"
    output += f"Original Dataset Length: {original_num_rows} rows, {original_num_columns} columns\n"
    output += f"Rows dropped due to missing values: {num_rows_dropped}\n"
    output += f"Dataset Length after dropping rows: {num_rows_after_dropping} rows, {num_columns} columns\n"
    output += "\nDataset (First {} Rows):\n{}\n\n\n".format(NUM_TOP_FEATURES, balance_data.head(NUM_TOP_FEATURES))

    X = balance_data.iloc[:, 0:NUM_FEATURES].values
    Y = [balance_data.loc[:, column].values for column in Y_COLUMNS]

    Y_encoded = encode_targets(Y)

    classifiers = []
    y_pred_en_list = []
    accuracy_scores = []
    top_feature_values = []

    output += "=" * 50 + "\n"

    isolated_rows = {}
    for column in Y_COLUMNS:
        isolated_rows[column] = balance_data[balance_data[column] > 0].copy()

    for i in range(len(Y_encoded)):
        output += "-" * 50 + "\n"

        X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded[i], test_size=0.3, random_state=100)

        clf_rf = train_random_forest(X_train, y_train)
        classifiers.append(clf_rf)
        y_pred = clf_rf.predict(X_test)
        y_pred_en_list.append(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)

        top_columns = get_top_features(clf_rf, balance_data.columns[:NUM_FEATURES], num_top_features=NUM_TOP_FEATURES)
        top_feature_values.append(get_top_feature_values(clf_rf, balance_data.columns[:NUM_FEATURES],
                                                        isolated_rows[Y_COLUMNS[i]].iloc[:, :NUM_FEATURES].values,
                                                        num_values=NUM_TOP_FEATURE_VALUES))

        output += f"Target: {Y_COLUMNS[i]}\n"
        output += f"Predicted values: {y_pred_en_list[i]}\n"
        output += f"Accuracy is {accuracy_scores[i] * 100:.2f}%\n"
        output += "Top {} X-columns (in descending order of importance):\n".format(NUM_TOP_FEATURES)
        for idx, column in enumerate(top_columns, 1):
            output += f"{idx}. {column}\n"
        output += "\n"

        pdf_path = os.path.join(OUTPUT_FOLDER, f'Decision_Tree_{Y_COLUMNS[i]}.pdf')
        save_decision_tree(clf_rf, balance_data.columns[:NUM_FEATURES], np.unique(Y[i]), pdf_path)

        output += f"Top {NUM_TOP_FEATURE_VALUES} feature values for the Top {NUM_TOP_FEATURES} X-columns ({Y_COLUMNS[i]} > 0):\n"
        if i < len(top_feature_values):
            for column in top_columns:
                values = top_feature_values[i][column]
                output += f"{column}:\n"
                for value, count in values.head(NUM_TOP_FEATURE_VALUES).items():
                    output += f"{value}: {count}\n"
                output += "\n"
        else:
            output += "No top feature values available.\n"
            output += "\n"

    output += "=" * 50 + "\n"

    output_path = os.path.join(OUTPUT_FOLDER, 'Output.pdf')
    save_output_to_pdf(output, output_path)

    logging.info("Output saved as PDF")


if __name__ == '__main__':
    main()
