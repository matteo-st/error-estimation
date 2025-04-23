import torch
from torch.distributions import MultivariateNormal


class BayesDetector:
    def __init__(self, classifier, weights, means, stds, n_classes, device=torch.device('cpu')):
        """
        Args:
            classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
            weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
            means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            n_cluster (int): Number of clusters to partition the error probability into.
            alpha (float): Confidence level parameter for interval widths.
            method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
            seed (int): Random seed for data generation.
            device (torch.device): Device on which to run the classifier.
        """
        self.classifier = classifier
        self.weights = weights      # torch.Tensor, shape: [n_classes]
        self.means = means          # torch.Tensor, shape: [n_classes, dim]
        self.stds = stds            # torch.Tensor, shape: [n_classes, dim]
        self.covs = torch.diag_embed(stds ** 2)
        self.n_classes = n_classes
        self.device = device

        # Initilize the density function
        self.pdfs = [MultivariateNormal(loc=self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_classes)]


    def func_proba_error(self, x):
        """
        Compute an error probability for a given input x.
        A simple proxy is: error probability = 1 - max(predicted probability)
        
        Args:
            x (np.ndarray): A 1D NumPy array of shape [dim].
            classifier (nn.Module): The classifier.
            device (torch.device): Device to use.
            
        Returns:
            proba_error (float): The error probability.
        """
        with torch.no_grad():
            # classifier returns (logits, probs)
            _, model_probs = self.classifier(x)
            model_pred = torch.argmax(model_probs, dim=1, keepdim=True)
        data_probs = [self.weights[i] *  torch.exp(self.pdfs[i].log_prob(x)) for i in range(self.n_classes)]
 
        data_probs = torch.stack(data_probs, dim=1) # [batch_size, n_classes]
        pred_prob = data_probs.gather(1, model_pred) # [batch_size, 1]
        # Normalize the probabilities
        return 1 - pred_prob / data_probs.sum(dim=1, keepdim=True)  # [batch_size, n_classes]


    def __call__(self, x):
    
        return self.func_proba_error(x)
    
def gini(logits, temperature=1.0, normalize=False):
    g = torch.sum(torch.softmax(logits / temperature, dim=1) ** 2, dim=1, keepdim=True)
    if normalize:
        return  (1 - g) / g 
    else:
        return 1 - g

class GiniDetector:
    def __init__(self, classifier, temperature, normalize, device=torch.device('cpu')):
        """
        Args:
            classifier (nn.Module): A PyTorch model that takes an input tensor of shape [1, dim] and returns (logits, probs).
            weights (torch.Tensor): Tensor of shape [n_classes] (e.g., [7]).
            means (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            stds (torch.Tensor): Tensor of shape [n_classes, dim] (e.g., [7, 10]).
            n_cluster (int): Number of clusters to partition the error probability into.
            alpha (float): Confidence level parameter for interval widths.
            method (str): The method to compute the cluster. (Currently only "uniform" is supported.)
            seed (int): Random seed for data generation.
            device (torch.device): Device on which to run the classifier.
        """
        self.classifier = classifier
        self.temperature = temperature
        self.device = device
        self.normalize = normalize



    def __call__(self, x):
        logits, _ = self.classifier(x)
        return gini(logits, temperature=self.temperature, normalize=self.normalize)