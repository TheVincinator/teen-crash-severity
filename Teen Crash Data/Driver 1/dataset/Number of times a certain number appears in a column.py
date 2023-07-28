import csv

def count_value_in_column(csv_file_path, column_name, target_value):
    count = 0
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[column_name] == target_value:
                count += 1
    return count

csv_file_path = 'remaining_rows_K_only.csv'
column_name = 'K_PEOPLE'
target_value = '1'

count = count_value_in_column(csv_file_path, column_name, target_value)
print(f"The value '{target_value}' appears {count} times in the column '{column_name}'.")
