import torch

class SimpleGenerator:
    def __init__(self, cond_dims):
        self.cond_dims = cond_dims
        self.total_dim = sum(cond_dims)  # Total length of concatenated one-hot vectors

    def generate(self, batch_size=1):
        # List to store one-hot encoded tensors for each dimension
        one_hot_tensors = []
        
        # Generate multinomial samples separately for each cond_dim
        for dim in self.cond_dims:
            # Create uniform probabilities for the current dimension
            probs = torch.ones(batch_size, dim) / dim
            # Sample one index per sample for this dimension
            indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # Shape: (batch_size,)
            # Convert indices to one-hot encoding
            one_hot = torch.zeros(batch_size, dim)
            one_hot.scatter_(1, indices.unsqueeze(-1), 1.0)
            one_hot_tensors.append(one_hot)
        
        # Concatenate all one-hot tensors along the feature dimension
        results = torch.cat(one_hot_tensors, dim=1)
        return results.unsqueeze(1)

# Test code
def test_simple_generator():
    # Initialize generator
    cond_dims = [2, 3]
    generator = SimpleGenerator(cond_dims=cond_dims)
    
    # Test parameters
    batch_size = 5
    
    # Generate samples
    samples = generator.generate(batch_size=batch_size)
    
    # Test 1: Check output shape
    expected_shape = (batch_size, sum(cond_dims))
    assert samples.shape == expected_shape, f"Expected shape {expected_shape}, got {samples.shape}"
    print(f"Test 1 Passed: Output shape is {samples.shape}")
    
    # Test 2: Check one-hot encoding for each condition
    for i, dim in enumerate(cond_dims):
        start_idx = sum(cond_dims[:i])
        end_idx = start_idx + dim
        segment = samples[:, start_idx:end_idx]
        # Each segment should have exactly one '1' per sample
        assert torch.all(segment.sum(dim=1) == 1), f"Segment {i} is not one-hot encoded"
        # Each segment should only contain 0s and 1s
        assert torch.all((segment == 0) | (segment == 1)), f"Segment {i} contains invalid values"
    print("Test 2 Passed: All segments are valid one-hot encodings")
    
    # Test 3: Check if samples are on correct device
    assert samples.device == torch.device(device), f"Expected device {device}, got {samples.device}"
    print(f"Test 3 Passed: Samples are on {samples.device}")
    
    # Print a sample for inspection
    print("\nExample output:")
    print(samples)

# Run the test
if __name__ == "__main__":
    test_simple_generator()