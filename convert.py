import csv
import pandas as pd

with open('phpKo8OWT.arff', 'r') as arff_file, open('contact-lenses.csv', 'w', newline='') as csv_file:
    # Read the ARFF file line by line
    lines = arff_file.readlines()

    # Create a CSV writer object
    writer = csv.writer(csv_file)

    # Write the header row to the CSV file
    header = [x.split()[1] for x in lines if x.startswith('@attribute')]
    writer.writerow(header)

    # Write the data rows to the CSV file
    data_start = lines.index('@data\n') + 1
    for line in lines[data_start:]:
        row = line.strip().split(',')
        writer.writerow(row)