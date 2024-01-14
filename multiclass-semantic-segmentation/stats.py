import os
import json
import numpy as np
import matplotlib.pyplot as plt


def read_json_files(directory: str) -> list:
    """
    Reads and parses JSON files from the specified directory.

    Args:
        - directory (str): The path to the directory containing JSON files.

    Returns:
        - list: A list of parsed JSON objects from the files in the directory.
    """
    results_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                results_list.append(data)
    return results_list


def plot_architecture_comparison(results_list: list) -> None:
    """
    Plots a visual comparison of architecture results.

    Args:
        - results_list (list): A list containing architecture results to be compared.

    Returns:
        - None.
    """
    # extract data for plotting
    model_names = [result["model_name"] for result in results_list]
    mIoU_values = [result["model_mIoU"] for result in results_list]
    mDice_values = [result["model_mDice"] for result in results_list]
    recall_values = [result["model_macro_recall"] for result in results_list]
    cpu_inference_times = [result["mean_inference_time"] for result in results_list if result["device"] == "cpu"]
    gpu_inference_times = [result["mean_inference_time"] for result in results_list if result["device"] == "gpu"]

    mIoU_classes_values = [result["model_mIoU_classes"] for result in results_list]
    dice_classes_values = [result["model_mDice_classes"] for result in results_list]
    recall_classes_values = [result["model_recall_classes"] for result in results_list]
    class_labels = ["background", "foliage", "waste"]
    # class_labels = [f"Class {i+1}" for i in range(len(mIoU_classes_values[0]))]

    # define a figure for plotting
    plt.figure(figsize=(18, 12))

    # plot mIoU values
    plt.subplot(1, 4, 1)
    if mIoU_values:
        plt.bar(model_names, mIoU_values, color="skyblue")
        plt.title("mIoU Comparison")
        plt.xlabel("Model name")
        plt.ylabel("mIoU")

    # plot mDice values
    plt.subplot(1, 4, 2)
    if mDice_values:
        plt.bar(model_names, mDice_values, color="lightgreen")
        plt.title("mDice Comparison")
        plt.xlabel("Model name")
        plt.ylabel("mDice")

    # plot recall values
    plt.subplot(1, 4, 3)
    if recall_values:
        plt.bar(model_names, recall_values, color="lightcoral")
        plt.title("Macro Recall Comparison")
        plt.xlabel("Model name")
        plt.ylabel("Macro Recall")

    # plot CPU/GPU inference times
    plt.subplot(1, 4, 4)
    if cpu_inference_times:
        plt.bar(model_names, cpu_inference_times, color="orange")
        plt.title("CPU inference time comparison")
        plt.xlabel("Model name")
        plt.ylabel("Mean Inference Time (ms)")
    else:
        plt.bar(model_names, gpu_inference_times, color="purple")
        plt.title("GPU inference time comparison")
        plt.xlabel("Model name")
        plt.ylabel("Mean Inference Time (ms)")
    
    plt.tight_layout()
    plt.show()

    # define a figure for plotting
    plt.figure(figsize=(18, 12))

    bar_width = 0.2

    # plot IoU for classes
    plt.subplot(1, 3, 1)
    for i, model_name in enumerate(model_names):
        positions = np.arange(len(class_labels)) + i * bar_width
        plt.bar(positions, mIoU_classes_values[i], width=bar_width, label=model_name)
    plt.xlabel("Class")
    plt.ylabel("mIoU values")
    plt.title("Comparison of different architectures")
    plt.xticks(np.arange(len(class_labels)) + (len(model_names) - 1) * bar_width / 2, class_labels)
    plt.legend()

    # plot Dice score for classes
    plt.subplot(1, 3, 2)
    for i, model_name in enumerate(model_names):
        positions = np.arange(len(class_labels)) + i * bar_width
        plt.bar(positions, dice_classes_values[i], width=bar_width, label=model_name)
    plt.xlabel("Class")
    plt.ylabel("Dice score values")
    plt.title("Comparison of different architectures")
    plt.xticks(np.arange(len(class_labels)) + (len(model_names) - 1) * bar_width / 2, class_labels)
    plt.legend()

    # plot Recall for classes
    plt.subplot(1, 3, 3)
    for i, model_name in enumerate(model_names):
        positions = np.arange(len(class_labels)) + i * bar_width
        plt.bar(positions, recall_classes_values[i], width=bar_width, label=model_name)
    plt.xlabel("Class")
    plt.ylabel("Recall values")
    plt.title("Comparison of different architectures")
    plt.xticks(np.arange(len(class_labels)) + (len(model_names) - 1) * bar_width / 2, class_labels)
    plt.legend()

    plt.tight_layout()
    plt.show()


# main script
if __name__ == "__main__":
    # replace this with the actual directory containing your JSON files
    json_directory = "./"
    results_list = read_json_files(json_directory)

    if results_list:
        plot_architecture_comparison(results_list)
    else:
        print("No JSON files found in the specified directory.")
