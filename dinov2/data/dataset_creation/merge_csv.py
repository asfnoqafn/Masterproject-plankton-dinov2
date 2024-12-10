import csv

def merge_csv_files(file1, file2, output_file):
    # Open the first file and skip the first line
    with open(file1, 'r') as f1:
        reader1 = csv.reader(f1)
        header = next(reader1)  # Skip the header
        data1 = list(reader1)

    # Open the second file and skip the first line
    with open(file2, 'r') as f2:
        reader2 = csv.reader(f2)
        next(reader2)  # Skip the header
        data2 = list(reader2)

    # Merge the data
    merged_data = data1 + data2

    # Write the merged data to the output file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(merged_data)


if __name__ == '__main__':

    # Example usage
    file1 = '/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp/training.csv'
    file2 = '/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp/validation.csv'
    output_file = '/home/hk-project-p0021769/hgf_vwg6996/data/seanoe_uvp/test.csv'

    merge_csv_files(file1, file2, output_file)
