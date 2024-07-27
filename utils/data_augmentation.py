import numpy as np

def rotate_grid(grid, k):
    return np.rot90(grid, k)

def flip_grid(grid, axis):
    return np.flip(grid, axis)

def mask_grid(grid, mask_ratio=0.1):
    mask = np.random.choice([0, 1], size=grid.shape, p=[mask_ratio, 1-mask_ratio])
    return grid * mask

def augment_data(input_grid, output_grid):
    # Randomly choose an augmentation
    aug_type = np.random.choice(['rotate', 'flip', 'mask', 'none'])
    
    if aug_type == 'rotate':
        k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
        input_grid = rotate_grid(input_grid, k)
        output_grid = rotate_grid(output_grid, k)
    elif aug_type == 'flip':
        axis = np.random.randint(0, 2)  # Horizontal or vertical flip
        input_grid = flip_grid(input_grid, axis)
        output_grid = flip_grid(output_grid, axis)
    elif aug_type == 'mask':
        input_grid = mask_grid(input_grid)
    
    return input_grid, output_grid