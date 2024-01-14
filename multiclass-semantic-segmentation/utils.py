import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# fix shape batch

def calculate_class_weights(dataloader: DataLoader, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights based on inverse class frequencies in a dataset.

    Parameters:
        - dataloader (DataLoader): The DataLoader containing the dataset.
        - num_classes (int): The number of classes in the dataset.

    Returns:
        - torch.Tensor: A tensor containing the calculated class weights.
    """
    # initialize a tensor to store the count of samples for each class
    class_counts = torch.zeros(num_classes)

    # calculate class frequencies
    for _, masks in dataloader:
        for class_idx in range(num_classes):
            class_counts[class_idx] += torch.sum(masks == class_idx).item()

    # calculate inverse class frequencies, avoiding division by zero
    inverse_class_frequencies = torch.where(class_counts > 0, 1 / class_counts, 0)

    # normalize weights
    weights = inverse_class_frequencies / inverse_class_frequencies.sum()

    return weights


def measure_performance_cpu(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """
    Measure the performance of the model on the CPU.

    Args:
        - model (nn.Module): The neural network model to evaluate.
        - test_loader (DataLoader): The DataLoader for the test dataset.
        - device (torch.device): The device on which to perform inference ("cpu").

    Returns:
        - tuple of float: A tuple containing the total inference time (in seconds) and any additional metrics as floats.

    Credits:
        - https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    """
    print(f"\nStarting measure inference-time on CPU...")
    
    dummy_input, _ = next(iter(test_loader))
    dummy_input = dummy_input.to(device)

    repetitions = 300
    timings = np.zeros((repetitions, 1))

    # cpu-warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # measure performance
    model.eval()
    with torch.no_grad():
        for rep in range(repetitions):
            # start to record
            start = time.time()
            # operation to measure
            _ = model(dummy_input)
            # end to record
            end = time.time()
            # compute inference time
            elapsed_time = end - start
            timings[rep] = elapsed_time * 1000.0

    mean_inf = (np.sum(timings) / repetitions)
    std_inf = np.std(timings)

    print(f"\nmean_inference_time: {mean_inf:.3f} ms ±{std_inf:.3f} ms.")
                    
    print(f"\nCPU measure performance Done...")

    return mean_inf, std_inf


def measure_performance_gpu(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """
    Measure the performance of the model.

    Args:
        - model (nn.Module): The neural network model to evaluate.
        - test_loader (DataLoader): The DataLoader for the test dataset.
        - device (torch.device): The device on which to perform inference ("cuda").

    Returns:
        - tuple of float: A tuple containing the total inference time (in seconds) and any additional metrics as floats.

    Credits:
        - https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    """
    print(f"\nStarting measure inference-time on GPU...")
    
    dummy_input, _ = next(iter(test_loader))
    dummy_input = dummy_input.to(device)

    # init loggers        
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    repetitions = 300
    timings = np.zeros((repetitions, 1))

    # gpu-warm-up
    for _ in range(10):
        _ = model(dummy_input)
    
    # measure performance
    model.eval()
    with torch.no_grad():
        for rep in range(repetitions):
            # start to record
            start.record()
            # operation to measure
            _ = model(dummy_input)
            # end to record
            end.record()                
            # wait for gpu sync
            torch.cuda.synchronize()
            # compute inference time                
            elapsed_time = start.elapsed_time(end)                
            timings[rep] = elapsed_time

    mean_inf = (np.sum(timings) / repetitions)
    std_inf = np.std(timings)

    print(f"\nStarting to measure throughput on GPU...")

    repetitions = 100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    throughput =   (repetitions * dummy_input.size(0)) / total_time
    
    print(f"\nmean_inference_time: {mean_inf:.3f} ms ±{std_inf:.3f} ms.")
    
    print("\nFinal throughput:", throughput)

    print(f"\nGPU measure performance Done...")

    return mean_inf, std_inf, throughput


def measure_transferring_time(device: torch.device, num_iterations: int = 300, shape: tuple = (1, 1, 416, 608)) -> tuple[float, float]:
    """ 
    Measure the transferring time taken to move images from host (CPU) to device (GPU). 

    Args:
        - device (torch.device): The target device, typically GPU.
        - num_iterations (int, optional): Number of iterations for measurement. Default is 300.
        - shape (tuple, optional): The shape of the images to be transferred. Default is (1, 1, 288, 216).

    Returns:
        - tuple[float, float]: A tuple containing the average transferring time and the standard deviation.
    """
    print("\nStarting to measure transferring time (host-cpu to device-gpu)...")

    transfer_times = []

    for _ in range(num_iterations):
        # create a random tensor on the host
        tensor_host = torch.randn(shape)

        # move the tensor to the device (GPU) and measure time
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        # record start event
        start_time.record()

        # move tensor to the device (GPU)
        tensor_host = tensor_host.to(device)

        # record end event
        end_time.record()
        torch.cuda.synchronize()

        # calculate the transfer time in milliseconds
        transfer_time_ms = start_time.elapsed_time(end_time)
        transfer_times.append(transfer_time_ms)

    mean_transfer_time = sum(transfer_times) / num_iterations
    std_transfer_time = np.std(transfer_times)

    print(f"\nMean time taken to transfer tensor to {device}: {mean_transfer_time:.3f} ms...")
    print(f"Standard Deviation: {std_transfer_time:.3f} ms")

    print("\nMeasure transferring time Done...\n")

    return mean_transfer_time, std_transfer_time
