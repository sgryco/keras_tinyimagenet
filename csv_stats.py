"""Extract statistics from all the csv files generated from extract_scalars."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import os
import sys


def main():
    tag_names = ('val_acc',)
    output_dir = 'csv_output'
    files = os.listdir(output_dir)
    summary_file = "summary.csv"
    summary_path = os.path.join(output_dir, summary_file)

    if os.path.exists(summary_path):
        print("Summary file already exists, exiting.")
        sys.exit(1)

    with open(summary_path, "w") as summ:
        csvsum = csv.writer(summ, delimiter=',')
        csvsum.writerow(["run", "max_val_acc", "rel_time", "epochs"])

        print("Loading data...")
        for f in files:
            if f == summary_file or not f.endswith("val_acc.csv"):
                continue
            with open(os.path.join(output_dir, f), 'r') as mycsv:
                csvreader = csv.reader(mycsv, delimiter=',')
                header = next(csvreader)
                if len(header) != 3:
                    raise ValueError("Wrong header length")

                maxacc = 0.
                init_time = -1.
                max_time = 0.
                max_epochs = 0

                for line in csvreader:
                    if init_time == -1.:
                        init_time = float(line[0])
                    acc = float(line[2])
                    if maxacc < acc:
                        maxacc = acc
                        max_time = float(line[0])
                        max_epochs = int(line[1])

                csvsum.writerow([f, maxacc, max_time - init_time, max_epochs])

        print("Done.")


if __name__ == '__main__':
    main()
