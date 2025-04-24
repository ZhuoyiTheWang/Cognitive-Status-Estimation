import csv
import sys

def compare_csv_files(file1, file2):
    differences = []
    with open(file1, newline='') as csvfile1, open(file2, newline='') as csvfile2:
        reader1 = csv.reader(csvfile1)
        reader2 = csv.reader(csvfile2)
        row_num = 1
        for row1, row2 in zip(reader1, reader2):
            if row1 != row2:
                differences.append(f"Row {row_num} differs:")
                differences.append(f"File 1: {row1}")
                differences.append(f"File 2: {row2}")
            row_num += 1
        # Check for extra rows in either file
        extra_rows1 = list(reader1)
        extra_rows2 = list(reader2)
        if extra_rows1:
            for row in extra_rows1:
                differences.append(f"Extra row in {file1} at line {row_num}: {row}")
                row_num += 1
        if extra_rows2:
            for row in extra_rows2:
                differences.append(f"Extra row in {file2} at line {row_num}: {row}")
                row_num += 1

    if differences:
        file_number = file1.split('/')[2]
        file_number = file_number.split('.')[0]
        
        # Write indices of different rows to a file
        with open(f'data/Differences/{file_number}.txt', 'w') as f:
            for diff in differences:
                f.write(diff + '\n')
    else:
        print("The CSV files are identical.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_csvs.py file1.csv file2.csv")
        sys.exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_csv_files(file1, file2)