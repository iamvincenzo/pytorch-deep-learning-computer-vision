import os
import json
import numpy as np
import matplotlib.pyplot as plt

def read_json_file(file_path):
    """
    Read data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded data from the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def find_json_files(root_folder):
    """
    Find all JSON files recursively in a given root folder.

    Args:
        root_folder (str): The root folder to search for JSON files.

    Returns:
        list: A list of paths to JSON files.
    """
    json_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

def plot_mean_accuracy_std(files, save_path=None):
    """
    Plot the mean, min, and max accuracy across folds from JSON files.

    Args:
        files (list): A list of paths to JSON files.
        save_path (str, optional): The path to save the figure. If None, the figure is displayed.

    Returns:
        None
    """
    accuracies = []

    for file in files:
        data = read_json_file(file)
        accuracies.append([fold_data["model_acc"] for fold_data in data.values()])

    accuracies = np.array(accuracies)

    mean_accuracy = np.mean(accuracies, axis=1)
    std_accuracy = np.std(accuracies, axis=1)
    min_accuracy = np.min(accuracies, axis=1)
    max_accuracy = np.max(accuracies, axis=1)

    # print("Accuracies:")
    # print(accuracies)
    # print()
    # print("Mean Accuracy:")
    # print(mean_accuracy)
    # print()
    # print("Standard Deviation:")
    # print(std_accuracy)
    # print()
    # print("Min Accuracy:")
    # print(min_accuracy)
    # print()
    # print("Max Accuracy:")
    # print(max_accuracy)

    # plotting
    plt.errorbar(range(1, len(mean_accuracy) + 1), mean_accuracy, yerr=std_accuracy, fmt="o-", label="Mean ±Std-Dev Accuracy")
    plt.scatter(range(1, len(min_accuracy) + 1), min_accuracy, color='red', marker='v', label="Min Accuracy")
    plt.scatter(range(1, len(max_accuracy) + 1), max_accuracy, color='green', marker='^', label="Max Accuracy")

    plt.title("Mean, Min, and Max Accuracy Across Folds")
    plt.xlabel("Scenario")
    plt.ylabel("Validation Accuracy")
    plt.legend()

    if save_path:
        plt.savefig(save_path, transparent=True)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    root_folder = "./statistic"

    # replace with the desired save path
    save_path = "figure_without_background.png"

    # find all JSON files recursively in the root folder
    json_files = find_json_files(root_folder)

    if json_files:
        plot_mean_accuracy_std(json_files, save_path=save_path)
    else:
        print("No JSON files found in the specified root folder.")
