import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(tn, fp, fn, tp):
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    fig = plt.figure()
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)

    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            value = confusion_matrix[i, j]
            color = "w" if value > confusion_matrix.max() / 2 else "k"
            plt.text(j, i, str(value), ha="center", va="center", color=color)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    # plt.show()

    return fig


def plot_roc(fpr, tpr, auc):
    fig = plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    # plt.show()

    return fig


def plot_prc(recall, precision):
    fig = plt.figure(figsize=(8, 8))

    plt.plot(recall, precision, color="darkorange", lw=2)
    plt.plot([0, 1], [1, 0], color="navy", lw=2, linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    # plt.legend(loc="lower right")
    # plt.show()

    return fig
