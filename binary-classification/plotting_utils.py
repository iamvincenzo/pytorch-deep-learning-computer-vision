import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_confusion_matrix(tn, fp, fn, tp):
    """
    Plot a confusion matrix.

    Args:
        tn (int): True Negative count.
        fp (int): False Positive count.
        fn (int): False Negative count.
        tp (int): True Positive count.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
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
    """
    Plot a Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (array-like): False Positive Rate.
        tpr (array-like): True Positive Rate.
        auc (float): Area Under the Curve (AUC) value.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
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
    """
    Plot a Precision-Recall Curve.

    Args:
        recall (array-like): Recall values.
        precision (array-like): Precision values.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    fig = plt.figure(figsize=(8, 8))

    plt.plot(recall, precision, color="darkorange", lw=2)
    plt.plot([0, 1], [1, 0], color="navy", lw=2, linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    # plt.legend(loc="lower right")
    # plt.show()

    return fig


# use hooks: 
# "hooks" are a mechanism that allows intercepting and recording 
# activities within a neural network module during data processing
conv_output = []

def append_conv(module, input, output):
    """
    Helper function used to visualize CNN activations: 
        the kernel emits high-response to some inputs.

    Parameters:
        module (torch.nn.Module): The convolutional layer module.
        input (torch.Tensor): The input tensor to the convolutional layer.
        output (torch.Tensor): The output tensor from the convolutional layer.

    Returns:
        None
    """
    # append all the conv layers and their respective wights to the list
    conv_output.append(output.detach().cpu()) 

def activations_viewer(model, img, max_filters_per_plot=64):
    """
    Function used to visualize CNN activations.

    Parameters:
        model (torch.nn.Module): The CNN model.
        img (torch.Tensor): The input image tensor.
        max_filters_per_plot (int, optional): The maximum number of filters to plot per layer. 
            Defaults to 64.

    Returns:
        None
    """
    # clear the list before each visualization
    global conv_output
    conv_output = []
    j = 0

    # check if the hooks have been registered before
    if not hasattr(activations_viewer, "hooks_registered"):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(append_conv)
        activations_viewer.hooks_registered = True      

    # forward pass
    out = model(img)

    # print filters shape
    print("\n\n")
    for c_out in conv_output:
        print(f"conv2-shape: {c_out.size()}")
    print("\n\n")

    # for each conv2d-layer of the model
    for num_layer in range(len(conv_output)):
        # get all the output of the convolution with filters with the input
        layer_viz = conv_output[num_layer][0, :max_filters_per_plot, :, :] # [0, :, :, :]
        layer_viz = layer_viz.data
        # number of filters in the layer
        num_filters = layer_viz.shape[0]
        # number of plots required
        num_plots = int(math.ceil(num_filters / max_filters_per_plot))

        # for each plot per filter
        for plot_num in range(num_plots):
            # calculate the range of filters to plot in the current plot
            start_idx = plot_num * max_filters_per_plot
            end_idx = (plot_num + 1) * max_filters_per_plot
            filters_to_plot = layer_viz[start_idx:end_idx]

            # create a new figure for the plot
            act_img_fig = plt.figure(figsize=(30, 30))
            # calculate the number of rows for the subplot grid based on the filters in the current plot
            num_rows = int(math.ceil(math.sqrt(filters_to_plot.shape[0])))

            # for each filter in the plot
            for i, filter in enumerate(filters_to_plot):
                # create a subplot for the filter
                plt.subplot(num_rows, num_rows, i + 1)
                # display the filter
                plt.imshow(filter)
                # set the title of the subplot to display the dimensions of the filter
                plt.title(f"{filter.shape[0]} x {filter.shape[1]}", fontsize="x-small")
                # turn off the axis labels
                plt.axis("off")

            # plt.show(block=False)
            # plt.pause(5)
            # plt.close()
            plt.show()
            
            # increment the counter for the figure number
            j += 1