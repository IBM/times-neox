#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse, os
from matplotlib.backends.backend_pdf import PdfPages
import zarr


def make_figs(group, pdf):
    ground_truth = group["ground_truth"][:]
    prediction = group["future"][:]
    past = group["past_target"][:, -context_length:]
    ground_truth_series = np.hstack([past, ground_truth])
    prediction_series = np.hstack([past, prediction])

    for gt, pr, gt_series, pr_series in zip(ground_truth, prediction, ground_truth_series, prediction_series): 
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

        ax2.plot(gt)
        ax2.plot(pr)
        ax1.plot(gt_series)
        ax1.plot(pr_series)
        ax1.axvline(context_length -1, color="r")
        fig.savefig(pdf, format='pdf') 
        
        plt.close(fig)


parser = argparse.ArgumentParser(
    prog='Time series visualization',
    description='Create PDF with predicted time series')

parser.add_argument(
    "-c", "--context_length", type = int, default = 100, 
    help = "length of the context to visualize")
parser.add_argument(
    "-i", "--input", type = str, required=True, 
    help = "Name of input zarr file")
parser.add_argument(
    "-p", "--pdf", type = str, required=True, 
    help = "Name of output PDF file.")
args = parser.parse_args()


context_length = args.context_length
input_filename = args.input
output_filename = args.pdf

if not os.path.exists(input_filename):
    raise FileExistsError(f"File {input_filename} does not exists, use --input option to set input file name.")

file = zarr.open(input_filename, "r")
pdf = PdfPages(output_filename)

for i in file.group_keys():
    make_figs(file[i], pdf)

pdf.close()
