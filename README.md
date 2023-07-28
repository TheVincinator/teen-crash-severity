# driver-crash-severity

This project was part of the VCU Summer Research Opportunities in Engineering program, focusing on "Leveraging Machine Learning Insights to Enhance Teenage Driver Safety and Road Awareness." The main objective of this research is to improve driver safety in Virginia by identifying the primary causes of injuries and fatalities among teenage drivers through comprehensive datasets and machine learning. By uncovering significant contributing factors, the project aims to propose targeted awareness programs and collaborative efforts to prevent car accidents and reduce road fatalities.

## Getting Started

To use this program, you can run it with the teen car crash data provided in the 'dataset' folders for both 'Driver 1' and 'Driver 2.' However, you also have the flexibility to apply the program to your own datasets. Simply follow the instructions below to configure the parameters and execute the 'decision_tree.py' script, which will generate PDF outputs of decision trees for you. Additionally, the code will create a summary file summarizing the accuracy of the decision tree classifier, the most important attributes, and the most frequently occurring feature values, among other details. To configure the parameters, you can edit the 'config.json' file.

### Configuring Parameters

The 'config.json' file contains the following parameters:

1. `file_path`: Specify the desired file path to your dataset. The default is set to the teen driver's data. Feel free to change this to your desired dataset path.

2. `tool`: Choose the type of decision tree output you prefer: "sklearn," "RapidMiner," or "both."

3. `num_features`: Select the number of columns you want to consider as attributes in your dataset. For example, if you set "num_features" to 30, the first 30 columns of your dataset will be treated as attributes.

### Sklearn Configurations

The following parameters are related to scikit-learn configurations:

4. `max_depth`: Set the maximum depth or levels in the decision tree. This helps control complexity and prevent overfitting.

5. `min_samples_leaf`: Determine the minimum samples required in a leaf node. It aids in regularizing the tree and improving generalization.

6. `random_state`: Set the random seed for reproducibility in the algorithm. It is useful for model comparison and debugging.

7. `Y_columns`: Choose the columns you want as targets (dependent variables). Please specify the columns in a list format, like this: ["column_1", "column_2", "column_3", "column_4"].

8. `summary_name`: Name your dataset in the 'output_summary.pdf' file. For example, if "summary_name" is "Teen Crashes," the text at the top of the 'output_summary.pdf' will be "Teen Crashes Dataset Summary."

9. `num_top_features`: Select the number of top features to display in the 'output_summary.pdf.' For instance, if you set the value to 5, the top five most important attributes in your dataset will be displayed.

10. `num_top_feature_values`: Configure the number of top feature values shown for the top features. If set to 5, the code will display the top five most frequently occurring values for the most important attributes in your dataset.

11. `display_all_tree_pdf`: Choose whether to show the decision tree classifier for all your target columns combined. If you have multiple target columns and want to see this decision tree, enter 'true.' The available options are 'true' or 'false'.

12. `test_size`: Set the test size for your decision tree. A value of 0.3 means that 30% of your data will be used for testing, while the remaining 70% will be used for training. Configure this value to suit your needs.

### RapidMiner Configurations

The following parameters are related to RapidMiner configurations:

**Note:** These configurations were tailored to the author's specific needs and dataset. If you do not require them, feel free to modify or remove this section of the code.

13. `remaining_rows_file_path`: Choose the file path to export a CSV file with dropped rows. This file can be imported into RapidMiner to run a decision tree classifier.

14. `target_columns_input`: Specify the target columns to facilitate working with data inside 'RapidMiner.' Please provide the columns without brackets or parenthesis, using quotation marks to close off the list (e.g., "column_1, column_2, column_3, column_4").

## VDOT Crash Data Dictionary

For your reference, the 'VDOT_Crash_Data_Dictionary_2022.pdf' file contains detailed information about each attribute and its corresponding values. You can use the crash dictionary to better understand the dataset attributes and their meanings.

Feel free to reach out if you have any questions or need further assistance with the program! Happy analyzing!
