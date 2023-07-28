# driver-crash-severity

This project was developed as part of the VCU Summer Research Opportunities in Engineering program. The research project, titled "Leveraging Machine Learning Insights to Enhance Teenage Driver Safety and Road Awareness," aims to improve driver safety in Virginia by identifying the primary causes of injury and death among teenage drivers using comprehensive datasets and machine learning techniques. By uncovering the significance of contributing factors, the project aims to propose targeted awareness programs and collaborative efforts to prevent car accidents and reduce fatalities on the roads.

## Getting Started

The program is designed to work with the provided teen car crash data located in the 'dataset' folders for both 'Driver 1' and 'Driver 2'. However, you can also use this program with your own datasets by configuring some parameters. Simply execute the 'decision_tree.py' script, and it will generate PDF outputs of decision trees for you. The code will also produce a summary file containing information such as the accuracy of the decision tree classifier, the most important attributes, and the most frequently occurring feature values.

## Configuration Parameters

### config.json Parameters

- `"file_path"`: Allows you to specify the file path to your dataset. The default file path points to the provided teen driver data. You can change this to the path of your desired dataset.

- `"tool"`: Specifies whether you want the code to output a 'sklearn' or 'RapidMiner' decision tree. The valid values are "sklearn", "RapidMiner", or "both".

- `"num_features"`: Lets you select the number of columns in your dataset that should be treated as attributes. For example, setting `"num_features": 30` means that the first 30 columns of your dataset will be considered attributes.

### sklearn Configurations

- `"max_depth"`: Sets the maximum depth or levels in the decision tree. This parameter controls the complexity of the tree and helps prevent overfitting.

- `"min_samples_leaf"`: Sets the minimum number of samples required in a leaf node of the decision tree. This regularization parameter improves generalization.

- `"random_state"`: Sets the random seed for reproducibility in the algorithm. It is useful for comparing models and debugging.

- `"Y_columns"`: Allows you to choose the columns that represent your target variables. Please provide the columns as a list, for example: `["column_1", "column_2", "column_3", "column_4"]`.

- `"summary_name"`: Lets you name your dataset in the 'output_summary.pdf' file. For instance, if you set `"summary_name": "Teen Crashes"`, the top of the 'output_summary.pdf' will display "Teen Crashes Dataset Summary:".

- `"num_top_features"`: Allows you to select the number of top features to display in the summary. For example, if you set `"num_top_features": 5`, the 'output_summary.pdf' will show the top five most important attributes in your dataset.

- `"num_top_feature_values"`: Lets you configure the number of top feature values shown for the selected top features. For instance, setting `"num_top_feature_values": 5` will display the top five most frequently occurring values for the most important attributes in your dataset.

- `"display_all_tree_pdf"`: Allows you to choose whether to show the decision tree classifier for all your target columns combined. If you have more than one target column and want to see this decision tree, set this parameter to `true`. Valid options are 'true' or 'false'.

- `"test_size"`: Lets you choose the test size for your decision tree. A value of 0.3 means that 30% of your data will be used for testing, and the remaining 70% will be used for training. Please configure this to suit your needs.

### RapidMiner Configurations

Please note that these configurations are specific to the developer's use case and may not be relevant to your dataset. You can disregard these configurations unless they align with your specific requirements. If you do not need the program to convert values or create separate CSV files, feel free to modify this section of the code as needed.

- `"remaining_rows_file_path"`: Allows you to specify the file path to export a CSV file containing dropped rows. This file can be imported into RapidMiner for further analysis.

- `"target_columns_input"`: Lets you specify the target columns for easier data handling within 'RapidMiner'. Please provide the column names as a comma-separated list without brackets or parentheses, for example: `"column_1, column_2, column_3, column_4"`. Use only quotation marks to enclose the list.

Feel free to use this program to analyze your own datasets and adjust the configurations in the `config.json` file according to your needs. Happy analyzing!
