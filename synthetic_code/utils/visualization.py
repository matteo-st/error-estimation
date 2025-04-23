import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter

def plot_decision_boundary(model, X, y, device='cpu', grid_points=200, sample_fraction=0.2,
                           title="Decision Boundary", save_path=None):
    """
    Plots the decision boundary of a PyTorch model on 2D data.
    
    To avoid overwhelming the plot with points, if X is large a random fraction
    (controlled by sample_fraction) is used for plotting.
    
    For binary classification, we smooth the probability difference to yield a
    cleaner boundary. For multi-class, we plot a filled contour of the predictions.
    Then, we overlay the data points; in addition, any point that has been wrongly
    classified is overplotted with a red border.
    
    Args:
        model (torch.nn.Module): Trained model that outputs logits.
        X (np.ndarray): 2D input data of shape (n_samples, 2).
        y (np.ndarray): Ground truth labels of shape (n_samples,).
        device (str): Device on which the model is (e.g., 'cpu' or 'cuda').
        grid_points (int): Number of points along each axis to form the grid.
        sample_fraction (float): Fraction of data points to plot if dataset is huge.
        title (str): Title for the plot.
        save_path (str, optional): If provided, the plot is saved to this file.
    """
    # Downsample data if necessary.
    if len(X) > 1000:
        idx = np.random.choice(len(X), int(len(X) * sample_fraction), replace=False)
        X_plot = X[idx]
        y_plot = y[idx]
    else:
        X_plot = X
        y_plot = y

    # Determine grid boundaries.
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
                         np.linspace(y_min, y_max, grid_points))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(grid_tensor)
        # If outputs are logits, compute probabilities.
        probs = torch.softmax(outputs, dim=1)
        _, _ = torch.max(outputs, dim=1)  # not used here directly.
    probs = probs.cpu().numpy()
    # In the binary case, compute a difference function.
    plt.figure(figsize=(8, 6))
    if probs.shape[1] == 2:
        diff = probs[:, 1] - probs[:, 0]
        diff = diff.reshape(xx.shape)
        # Smooth the difference using a Gaussian filter.
        diff_smooth = gaussian_filter(diff, sigma=1)
        # Plot the 0 contour (decision boundary) on the smoothed map.
        plt.contour(xx, yy, diff_smooth, levels=[0], linewidths=2, colors='k')
        plt.contourf(xx, yy, diff_smooth, alpha=0.3, cmap=plt.cm.Paired)
    else:
        # For multiclass, plot the predicted regions using the argmax.
        preds = np.argmax(probs, axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.Paired)
    
    # Compute model predictions on the (downsampled) data used for plotting.
    X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32, device=device)
    with torch.no_grad():
        outputs_plot = model(X_plot_tensor)
        model_preds = torch.argmax(torch.softmax(outputs_plot, dim=1), dim=1).cpu().numpy()
    
    # Plot all data points with class colors (with black edge).
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, s=40,
                          edgecolor='k', cmap=plt.cm.Paired, label="Data Points")
    
    # Identify misclassified points.
    misclassified_idx = model_preds != y_plot
    if np.any(misclassified_idx):
        plt.scatter(X_plot[misclassified_idx, 0], X_plot[misclassified_idx, 1],
                    facecolors='none', edgecolors='red', s=80, linewidths=1.5,
                    label="Misclassified")
    
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from scipy.ndimage import gaussian_filter

# def plot_decision_boundary(model, X, y, device='cpu', grid_points=200, sample_fraction=0.2,
#                            title="Decision Boundary", save_path=None):
#     """
#     Plots the decision boundary of a PyTorch model on 2D data.
    
#     To avoid overwhelming the plot with points, if X is large a random fraction
#     (controlled by sample_fraction) is used for plotting.
    
#     For binary classification, we smooth the probability difference to yield a
#     cleaner boundary. For multi-class, we plot a filled contour of the predictions.
    
#     Args:
#         model (torch.nn.Module): Trained model that outputs logits.
#         X (np.ndarray): 2D input data of shape (n_samples, 2).
#         y (np.ndarray): Ground truth labels of shape (n_samples,).
#         device (str): Device on which the model is (e.g., 'cpu' or 'cuda').
#         grid_points (int): Number of points along each axis to form the grid.
#         sample_fraction (float): Fraction of data points to plot if dataset is huge.
#         title (str): Title for the plot.
#         save_path (str, optional): If provided, the plot is saved to this file.
#     """
#     # Downsample the training data if needed.
#     if len(X) > 1000:
#         idx = np.random.choice(len(X), int(len(X) * sample_fraction), replace=False)
#         X_plot = X[idx]
#         y_plot = y[idx]
#     else:
#         X_plot = X
#         y_plot = y

#     # Determine grid boundaries.
#     x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
#     y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points),
#                          np.linspace(y_min, y_max, grid_points))
    
#     grid = np.c_[xx.ravel(), yy.ravel()]
#     grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    
#     model.eval()
#     with torch.no_grad():
#         outputs = model(grid_tensor)
#         # If outputs are logits, compute probabilities.
#         probs = torch.softmax(outputs, dim=1)
#         _, preds = torch.max(outputs, dim=1)
#     preds = preds.cpu().numpy().reshape(xx.shape)
#     probs = probs.cpu().numpy()
    
#     plt.figure(figsize=(8, 6))
    
#     # For binary classification, compute a smoothed difference map.
#     if probs.shape[1] == 2:
#         diff = probs[:, 1] - probs[:, 0]
#         diff = diff.reshape(xx.shape)
#         # Smooth the difference using a Gaussian filter.
#         # diff_smooth = gaussian_filter(diff, sigma=1)
#         diff_smooth = diff
#         # Plot the contour where diff_smooth==0.
#         CS = plt.contour(xx, yy, diff_smooth, levels=[0], linewidths=2, colors='k')
#         plt.contourf(xx, yy, diff_smooth, alpha=0.3, cmap=plt.cm.Paired)
#     else:
#         # For multi-class, simply plot the predicted regions.
#         plt.contourf(xx, yy, preds, alpha=0.3, cmap=plt.cm.Paired)
    
#     # Overlay the (possibly downsampled) data points.
#     plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, s=40, edgecolor='k', cmap=plt.cm.Paired)
#     plt.title(title)
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.grid(True)
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#     plt.show()
