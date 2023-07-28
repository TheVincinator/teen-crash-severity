import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pydotplus
import logging

# Constants
CONFIG_FILE_PATH = "config.json"
OUTPUT_FOLDER = "pdf_outputs"
DECISION_TREE_OUTPUTS = "decision_trees"
CSV_FILE_OUTPUTS = "csv_files"
RAPIDMINER_OUTPUT_FOLDER = "rapidminer_data"
PROCESSED_DATA = "processed_data"
TRAINED_DATA = "rapidminer_trained_data"
TEST_DATA = "rapidminer_test_data"


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


def train_decision_tree(X: np.ndarray, y: list) -> DecisionTreeClassifier:
    with open(CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)

    max_depth = config.get("max_depth", 3)
    min_samples_leaf = config.get("min_samples_leaf", 5)
    random_state = config.get("random_state", 100)

    clf_dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    clf_dt.fit(X, y)
    return clf_dt


def get_top_features(clf: DecisionTreeClassifier, feature_names: list, num_top_features: int) -> pd.Series:
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    top_columns = importance_df.sort_values(by='Importance', ascending=False).head(num_top_features)['Feature']
    return top_columns


def save_decision_tree(clf: DecisionTreeClassifier, feature_names: list, unique_values: np.ndarray,
                       output_path: str):
    dot_data = export_graphviz(clf, out_file=None, feature_names=feature_names,
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


def create_sklearn_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    decision_trees_folder = os.path.join(OUTPUT_FOLDER, DECISION_TREE_OUTPUTS)
    if not os.path.exists(decision_trees_folder):
        os.makedirs(decision_trees_folder)
    csv_files_folder = os.path.join(OUTPUT_FOLDER, CSV_FILE_OUTPUTS)
    if not os.path.exists(csv_files_folder):
        os.makedirs(csv_files_folder)


def configure_logging():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())


def get_top_feature_values(clf: DecisionTreeClassifier, feature_names: list, X_test: np.ndarray,
                           num_values: int) -> dict:
    top_feature_values = {}
    for feature_idx, feature_name in enumerate(feature_names):
        feature_values = X_test[:, feature_idx]
        value_counts = pd.Series(feature_values).value_counts()
        top_values = value_counts.head(num_values)
        top_feature_values[feature_name] = top_values
    return top_feature_values


def create_rapidminer_output_folder():
    if not os.path.exists(RAPIDMINER_OUTPUT_FOLDER):
        os.makedirs(RAPIDMINER_OUTPUT_FOLDER)
    processed_data_folder = os.path.join(RAPIDMINER_OUTPUT_FOLDER, PROCESSED_DATA)
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)
    trained_data_folder = os.path.join(RAPIDMINER_OUTPUT_FOLDER, TRAINED_DATA)
    if not os.path.exists(trained_data_folder):
        os.makedirs(trained_data_folder)
    test_data_folder = os.path.join(RAPIDMINER_OUTPUT_FOLDER, TEST_DATA)
    if not os.path.exists(test_data_folder):
        os.makedirs(test_data_folder)


def process_target_columns(df, selected_columns):
    for col in selected_columns:
        if col not in df.columns:
            print(f"Target column '{col}' not found in the DataFrame.")
            continue

        # Make a copy of the original DataFrame to keep the original values intact
        processed_df = df.copy()

        # Change all values > 0 in the selected target column to 1
        processed_df[col] = processed_df[col].apply(lambda x: 1 if x > 0 else x)

        # Remove other target columns from the DataFrame
        other_target_columns = [c for c in selected_columns if c != col]
        processed_df.drop(columns=other_target_columns, inplace=True)

        # Save the DataFrame to a new CSV file
        processed_csv_file_path = os.path.join(RAPIDMINER_OUTPUT_FOLDER, PROCESSED_DATA, f"processed_{col}.csv")
        processed_df.to_csv(processed_csv_file_path, index=False)
        logging.info(f"Processed data for '{col}' saved to: {processed_csv_file_path}")


def main():
    configure_logging()
    create_sklearn_output_folder()

    index_shifter_for_display_all_tree = 0

    # Read the config file
    with open(CONFIG_FILE_PATH) as config_file:
        config = json.load(config_file)
        NUM_FEATURES = config.get("num_features", "")

    # Check if the user wants to use 'sklearn', 'RapidMiner', or both
    tool_choice = config.get("tool", "both").lower()

    if tool_choice == "both" or tool_choice == "sklearn":

        Y_COLUMNS = config.get("Y_columns", "")
        SUMMARY_NAME = config.get("summary_name", "")
        NUM_TOP_FEATURES = config.get("num_top_features", 5)
        NUM_TOP_FEATURE_VALUES = config.get("num_top_feature_values", 5)
        DISPLAY_ALL_TREE_PDF = config.get("display_all_tree_pdf", False)
        TEST_SIZE = config.get("test_size", 0.3)
        RANDOM_STATE = config.get("random_state", 100)

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

        # Export the new list with dropped rows to a CSV file
        remaining_rows_path = os.path.join(OUTPUT_FOLDER, CSV_FILE_OUTPUTS, 'remaining_rows.csv')
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

        if DISPLAY_ALL_TREE_PDF:
            X_train, X_test, y_train, y_test = train_test_split(X, np.column_stack(Y_encoded), test_size=TEST_SIZE,
                                                                random_state=RANDOM_STATE)

            index_shifter_for_display_all_tree = 1
            clf_dt_all = train_decision_tree(X_train, y_train)
            classifiers.append(clf_dt_all)
            y_pred_all = clf_dt_all.predict(X_test)
            y_pred_en_list.append(y_pred_all)
            accuracy_all = accuracy_score(y_test, y_pred_all)
            accuracy_scores.append(accuracy_all)

            top_columns_all = get_top_features(clf_dt_all, balance_data.columns[:NUM_FEATURES],
                                               num_top_features=NUM_TOP_FEATURES)
            top_feature_values_all = get_top_feature_values(clf_dt_all, balance_data.columns[:NUM_FEATURES], X_test,
                                                            num_values=NUM_TOP_FEATURE_VALUES)

            output += "-" * 50 + "\n"
            output += "Target: All\n"
            output += f"Predicted values: {y_pred_all}\n"
            output += f"Accuracy is {accuracy_all * 100:.2f}%\n"
            output += "Top {} X-columns (in descending order of importance):\n".format(NUM_TOP_FEATURES)
            for idx, column in enumerate(top_columns_all, 1):
                output += f"{idx}. {column}\n"
            output += "\n"

            if DISPLAY_ALL_TREE_PDF:
                pdf_path = os.path.join(OUTPUT_FOLDER, DECISION_TREE_OUTPUTS, 'Decision_Tree_All.pdf')
                save_decision_tree(clf_dt_all, balance_data.columns[:NUM_FEATURES], np.unique(np.concatenate(Y)), pdf_path)

                output += f"Top {NUM_TOP_FEATURE_VALUES} feature values for the Top {NUM_TOP_FEATURES} X-columns (All > 0):\n"
                for column in top_columns_all:
                    values = top_feature_values_all[column]
                    output += f"{column}:\n"
                    for value, count in values.head(NUM_TOP_FEATURE_VALUES).items():
                        output += f"{value}: {count}\n"
                    output += "\n"

                output += "=" * 50 + "\n"

        isolated_rows = {}
        for column in Y_COLUMNS:
            isolated_rows[column] = balance_data[balance_data[column] > 0].copy()

        for i in range(len(Y_encoded)):
            output += "-" * 50 + "\n"

            X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded[i], test_size=TEST_SIZE, random_state=RANDOM_STATE)

            clf_dt = train_decision_tree(X_train, y_train)
            classifiers.append(clf_dt)
            y_pred = clf_dt.predict(X_test)
            y_pred_en_list.append(y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

            top_columns = get_top_features(clf_dt, balance_data.columns[:NUM_FEATURES], num_top_features=NUM_TOP_FEATURES)
            top_feature_values.append(get_top_feature_values(clf_dt, balance_data.columns[:NUM_FEATURES],
                                                             isolated_rows[Y_COLUMNS[i]].iloc[:, :NUM_FEATURES].values,
                                                             num_values=NUM_TOP_FEATURE_VALUES))

            output += f"Target: {Y_COLUMNS[i]}\n"
            output += f"Predicted values: {y_pred_en_list[i+index_shifter_for_display_all_tree]}\n"
            output += f"Accuracy is {accuracy_scores[i+index_shifter_for_display_all_tree] * 100:.2f}%\n"
            output += "Top {} X-columns (in descending order of importance):\n".format(NUM_TOP_FEATURES)
            for idx, column in enumerate(top_columns, 1):
                output += f"{idx}. {column}\n"
            output += "\n"

            pdf_path = os.path.join(OUTPUT_FOLDER, DECISION_TREE_OUTPUTS, f'Decision_Tree_{Y_COLUMNS[i]}.pdf')
            save_decision_tree(clf_dt, balance_data.columns[:NUM_FEATURES], np.unique(Y[i]), pdf_path)

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

        output_path = os.path.join(OUTPUT_FOLDER, 'output_summary.pdf')
        save_output_to_pdf(output, output_path)

        logging.info("Output saved as PDF")

        # Save trained dataset to a CSV file with both attribute and target columns
        trained_data_with_target = pd.DataFrame(X_train, columns=balance_data.columns[:NUM_FEATURES])
        for i, column in enumerate(Y_COLUMNS):
            trained_data_with_target[column] = original_data[column].values[:X_train.shape[0]]

        trained_data_with_target.to_csv(os.path.join(OUTPUT_FOLDER, CSV_FILE_OUTPUTS, f'trained_dataset_with_targets_{(1-TEST_SIZE) * 100}%.csv'), index=False)

        logging.info("Trained dataset with target columns saved as CSV")


    if tool_choice == "both" or tool_choice == "rapidminer":

        # Load the dataset
        file_path = get_file_path()

        # Drop rows with missing values
        balance_data = load_data(file_path, NUM_FEATURES)

        # Call create_rapidminer_output_folder to ensure the necessary folders exist
        create_rapidminer_output_folder()

        # Export the new list with dropped rows to a CSV file
        remaining_rows_path = os.path.join(RAPIDMINER_OUTPUT_FOLDER, PROCESSED_DATA, 'remaining_rows.csv')
        balance_data.to_csv(remaining_rows_path, index=False)

        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv('rapidminer_data/processed_data/remaining_rows.csv')
        except FileNotFoundError:
            print("File not found. Please check the file path and try again.")
        else:

            # Ask the user to input the target columns
            target_columns_input = config.get("target_columns_input", "")
            selected_target_columns = [col.strip() for col in target_columns_input.split(',')]

            invalid_columns = set(selected_target_columns) - set(data.columns)
            if invalid_columns:
                print(
                    f"Invalid column(s) provided: {', '.join(invalid_columns)}. Please choose valid target column(s).")
            else:
                # Process the data for each selected target column
                process_target_columns(data, selected_target_columns)


if __name__ == '__main__':
    main()
