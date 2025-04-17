import numpy as np
import torch
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

def calculate_vc_dimension(model, static_data, time_series_data, epochs=50):
    """
    Calculate an empirical approximation of VC Dimension for a given model.
    """
    # Shuffling labels to create a random dataset
    shuffled_static, shuffled_time_series = shuffle(static_data, time_series_data)
    
    # Calculating model accuracy on random data
    losses = []
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training on shuffled dataset
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(shuffled_static, shuffled_time_series)
        labels = torch.zeros_like(predictions)
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
    
    avg_loss = np.mean(losses)
    return avg_loss


def calculate_rademacher_complexity(model, static_data, time_series_data):
    """
    Calculate an empirical approximation of the Rademacher Complexity for a given model.
    """
    static_data = torch.Tensor(static_data)
    time_series_data = torch.Tensor(time_series_data)
    
    # Generate random Rademacher variables
    rademacher_vars = 2 * (np.random.binomial(1, 0.5, static_data.shape[0]) - 0.5)
    rademacher_vars = torch.Tensor(rademacher_vars)
    
    # Calculate output differences for original and perturbed datasets
    original_output = model(static_data, time_series_data)
    perturbed_static = static_data * rademacher_vars[:, None]
    perturbed_output = model(perturbed_static, time_series_data)
    
    # Calculate Rademacher Complexity
    complexity = torch.mean(perturbed_output - original_output)
    return complexity

# Assuming Bi_cross_trained is your trained model and the provided dataset
static_data = np.random.rand(6, 200, 512)  # Replace with your real data
time_series_data = np.random.rand(13, 200, 9, 512)  # Replace with your real data

vc_dimension = calculate_vc_dimension(Bi_cross_trained, static_data, time_series_data)
print(f"Empirical VC Dimension: {vc_dimension}")

rademacher_complexity = calculate_rademacher_complexity(Bi_cross_trained, static_data, time_series_data)
print(f"Empirical Rademacher Complexity: {rademacher_complexity}")
