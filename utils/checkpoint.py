import os
import torch
from typing import Dict, Any

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, score: float, filename: str):
    """
    Save model checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'score': score
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename: str) -> Dict[str, Any]:
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def save_nas_results(best_model: Dict[str, Any], median_model: Dict[str, Any], worst_model: Dict[str, Any], 
                     save_dir: str):
    """
    Save the best, median, and worst models from a NAS run.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(best_model, os.path.join(save_dir, 'best_model.pth'))
    torch.save(median_model, os.path.join(save_dir, 'median_model.pth'))
    torch.save(worst_model, os.path.join(save_dir, 'worst_model.pth'))

def load_nas_result(filename: str) -> Dict[str, Any]:
    """
    Load a saved NAS result.
    """
    return torch.load(filename)