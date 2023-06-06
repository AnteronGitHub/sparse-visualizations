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

def format_training_specs(training_specs):
    [epochs, batch_size, batches] = training_specs.split("_")
    return f"{epochs} epochs, {batch_size} batch size, {batches} batches"

def format_compression_specs(training_specs):
    [feature_compression_factor, resolution_compression_factor, depruneProps] = training_specs.split("_")
    return f"{feature_compression_factor} feature compression, {resolution_compression_factor} resolution compression, {depruneProps}"

def format_metric_label(metric):
    if metric == "bytes_sent":
        return "Bytes sent (GB)"
    else:
        return metric.replace("_", " ")

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
    plt.figure(figsize=(16,12))
    for filepath in filepaths:
        application, suite, node, model, dataset, training_specs, compression_specs, date, time = deserialize_sparse_benchmark_file_name(filepath)
        df = pd.read_csv(filepath, usecols=["timestamp", metric])
        xs = np.array(df[df.columns[0]]/60.0)
        if metric == "bytes_sent":
            ys = np.array(df[df.columns[1]]/1000.0/1000.0)
        else:
            ys = np.array(df[df.columns[1]])
        plt.plot(xs, ys, get_plot_color(suite), label=f"{suite} {time}")

    training_specs = format_training_specs(training_specs)
    compression_specs = format_compression_specs(compression_specs)
    metric_label = format_metric_label(metric)
    plt.xlabel('Time (min)')
    plt.ylabel(metric_label)
    plt.xlim(0)
    plt.ylim(0)
    plt.title(f"{model}/{dataset}, {training_specs}, {compression_specs}")
    plt.legend(loc='lower right')

    plt.savefig(f"{application}-{metric}-{date}.svg", dpi=400)
#    plt.show()

if __name__ == "__main__":
    benchmark_files = parse_arguments()
    plot_metric(benchmark_files)
    plot_metric(benchmark_files, metric="bytes_sent")
