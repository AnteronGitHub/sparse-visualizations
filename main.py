import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pathlib
import seaborn as sns

def parse_arguments():
    """Parses program arguments from environment variables.
    """
    benchmark_dir = os.environ.get("SPARSE_BENCHMARK_DIR")
    result_files = filter(lambda file: pathlib.Path(file).suffix == ".csv", os.listdir(benchmark_dir))

    return [benchmark_dir + "/" + file for file in result_files]

def deserialize_sparse_benchmark_file_name(filepath):
    """Deserializes experiment metadata from file name based on sparse benchmark convention.
    """
    filename = filepath.split("/")[-1]
    filename = filename.split(".")[0]

    [example, application, suite, node, model, dataset, batch_specs, additional_data, date, time, benchmark_id] = filename.split("-")

    return example, application, suite, node, model, dataset, batch_specs, additional_data, date, time, benchmark_id

def format_batch_specs(batch_specs):
    [batch_size, batches] = batch_specs.split("_")
    return f"{batch_size} batch size, {batches} batches"

def format_additional_data(example, additional_data):
    if example == "splitnn":
        return f"{additional_data} epochs"

    deprunePhases = ""
    for (i, entry) in enumerate(additional_data.split("_")):
        if i == 0:
            feature_compression_factor = entry
        elif i == 1:
            resolution_compression_factor = entry
        elif i % 2 == 0:
            deprunePhases += f", {entry} epochs"
        else:
            deprunePhases += f" with {entry} budget"
    return f"{feature_compression_factor} feature compression, {resolution_compression_factor} resolution compression{deprunePhases}"

def format_metric_label(metric):
    if metric == "bytes_sent":
        return "Bytes sent (MB)"
    else:
        return metric.replace("_", " ")

def get_plot_color(suite, example):
    if suite == "edge_offloading":
        return "b"
    elif suite == "edge_split":
        if example == "deprune":
            return "r"
        else:
            return "y"
    else:
        return "g"

def preprocess_results(df, metric):
    if metric != "samples_processed":
        return df

    # Linearly interpolate plot by dropping rows that do not increase the samples processed
    samples_processed = None
    res_rows = []
    for index, row in df.iterrows():
        if samples_processed is None or samples_processed != row['samples_processed']:
            samples_processed = row['samples_processed']
            res_rows.append(row)

    return pd.DataFrame(res_rows, columns=df.columns)

def get_plot_linewidth(metric):
    if metric == "samples_processed":
        return .7

    return 1.5

def plot_metric(filepaths, metric = "samples_processed"):
    """Plots samples processed column.
    """
    plt.figure(figsize=(16,12))

    title_additional_data = None
    title_additional_data_locked = False
    plotted_labels = []
    for filepath in filepaths:
        example, application, suite, node, model, dataset, batch_specs, additional_data, date, time, benchmark_id = deserialize_sparse_benchmark_file_name(filepath)
        label = f"{example} {suite}"
        linewidth = get_plot_linewidth(metric)
        if metric == "bytes_sent" and label in plotted_labels:
            continue    # Only one plot per suite, since they tend to use the same physical network interface.

        if not title_additional_data_locked:
            title_additional_data = format_additional_data(example, additional_data)
            if example == "deprune":
                title_additional_data_locked = True

        df = pd.read_csv(filepath, usecols=["timestamp", metric])
        df = preprocess_results(df, metric)
        xs = np.array(df[df.columns[0]]/60.0)
        if metric == "bytes_sent":
            ys = np.array(df[df.columns[1]]/1000.0/1000.0)
        else:
            ys = np.array(df[df.columns[1]])
        if label in plotted_labels:
            plt.plot(xs, ys, get_plot_color(suite, example), linewidth=linewidth)
        else:
            plt.plot(xs, ys, get_plot_color(suite, example), label=label, linewidth=linewidth)
            plotted_labels.append(label)

    batch_specs = format_batch_specs(batch_specs)
    metric_label = format_metric_label(metric)
    plt.xlabel('Time (min)')
    plt.ylabel(metric_label)
    plt.xlim(0)
    plt.ylim(0)
    plt.title(f"{model}/{dataset}, {batch_specs}, {title_additional_data}")
    plt.legend(loc='lower right')

    plt.savefig(f"{application}-{metric}-{date}.png", dpi=400)
#    plt.show()

def plot_boxplot():
    plt.figure(figsize=(8,4))

    df_fcfs = pd.read_csv('data/batching-experiments-1-datasources/splitnn-learning-edge_offloading-datasource0-VGG-CIFAR10-1_128-1-tasks-20230825-121444-ceb2336e_052e_4bf4_a84e_0c79133f2af2.csv', usecols=["processing_time"])
    df_batching = pd.read_csv('data/batching-experiments-1-datasources/splitnn-learning-edge_offloading-datasource0-VGG-CIFAR10-256_128-1-tasks-20230825-122505-271b3dc6_966b_4667_85ae_2183d9e63332.csv')

    result_dir = 'data/batching-experiments-32-datasources_fcfs'
    df_fcfs_32 = pd.concat([pd.read_csv(f"{result_dir}/{file}") for file in os.listdir(result_dir)])

    result_dir = 'data/batching-experiments-32-datasources_batched'
    df_batching_32 = pd.concat([pd.read_csv(f"{result_dir}/{file}") for file in os.listdir(result_dir)])

    result_dir = 'data/batching-experiments-256-datasources_fcfs'
    df_fcfs_256 = pd.concat([pd.read_csv(f"{result_dir}/{file}") for file in os.listdir(result_dir)])

    result_dir = 'data/batching-experiments-256-datasources_batched'
    df_batching_256 = pd.concat([pd.read_csv(f"{result_dir}/{file}") for file in os.listdir(result_dir)])

    df_fcfs = df_fcfs.assign(no_datasources=1, policy='FCFS')
    df_batching = df_batching.assign(no_datasources=1, policy='batching')
    df_fcfs_32 = df_fcfs_32.assign(no_datasources=32, policy='FCFS')
    df_batching_32 = df_batching_32.assign(no_datasources=32, policy='batching')
    df_fcfs_256 = df_fcfs_256.assign(no_datasources=256, policy='FCFS')
    df_batching_256 = df_batching_256.assign(no_datasources=256, policy='batching')

    df = pd.concat([df_fcfs, df_batching, df_fcfs_32, df_batching_32, df_fcfs_256, df_batching_256])
    ax = sns.boxplot(x="no_datasources", y="processing_time", hue="policy", data=df, color="grey")

    plt.savefig(f"batching_experiments_20230825.png", dpi=400)

if __name__ == "__main__":
    benchmark_files = parse_arguments()
    plot_boxplot()
    #plot_metric(benchmark_files)
    #plot_metric(benchmark_files, metric="bytes_sent")
