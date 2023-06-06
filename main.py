import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def parse_arguments():
    """Parses program arguments from environment variables.
    """
    benchmark_files = os.environ.get("SPARSE_BENCHMARK_FILE")

    return benchmark_files.split("|")

def deserialize_sparse_benchmark_file_name(filepath):
    """Deserializes experiment metadata from file name based on sparse benchmark convention.
    """
    filename = filepath.split("/")[-1]
    filename = filename.split(".")[0]

    [application, suite, node, model, dataset, training_specs, compression_specs, date, time] = filename.split("-")

    return application, suite, node, model, dataset, training_specs, compression_specs, date, time

def get_plot_color(suite):
    if suite == "edge_offloading":
        return "b"
    elif suite == "edge_split":
        return "r"
    else:
        return "g"

def plot_metric(filepaths, metric = "samples_processed"):
    """Plots samples processed column.
    """
    for filepath in filepaths:
        application, suite, node, model, dataset, training_specs, compression_specs, date, time = deserialize_sparse_benchmark_file_name(filepath)
        df = pd.read_csv(filepath, usecols=["timestamp", metric])
        xs = np.array(df[df.columns[0]])
        ys = np.array(df[df.columns[1]])
        plt.plot(xs, ys, get_plot_color(suite), label=f"{suite}-{time}")

    plt.xlabel('Time (s)')
    plt.ylabel(metric)
    plt.xlim(0)
    plt.ylim(0)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    benchmark_files = parse_arguments()
    plot_metric(benchmark_files)
    plot_metric(benchmark_files, metric="bytes_sent")
