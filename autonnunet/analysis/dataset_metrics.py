import numpy as np


def compute_compactness(class_mask: np.ndarray) -> float:
    volume = np.sum(class_mask)
    if volume == 0:
        return 0.
    
    # We need the centroid to compute the radius of the enclosing sphere
    coordinates = np.argwhere(class_mask)  
    centroid = np.mean(coordinates, axis=0)
    
    # Now we can compute the radius of the enclosing sphere
    distances = np.linalg.norm(coordinates - centroid, axis=1)
    radius = np.max(distances)
    
    if radius == 0:  
        return float('inf')
    compactness = volume / ((4 / 3) * np.pi * radius**3)
    
    return compactness


def compute_std_per_axis(class_mask: np.ndarray) -> float:
    if class_mask.ndim == 1:
        return float(np.std(class_mask))
    elif class_mask.ndim == 2:
        return float(np.std(class_mask, axis=(0, 1)))
    elif class_mask.ndim == 3:
        return float(np.std(class_mask, axis=(0, 1, 2)))
    else:
        raise ValueError("Array must have at most 4 dimensions")