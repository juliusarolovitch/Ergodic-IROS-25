import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from mapGenerator import mapGenerator

class DirichletGMM:
    def __init__(self, max_components=10, weight_threshold=1e-3, plot_gaussians=False):
        self.max_components = max_components
        self.weight_threshold = weight_threshold
        self.plot_gaussians = plot_gaussians
        self.model = BayesianGaussianMixture(n_components=self.max_components, 
                                             weight_concentration_prior_type='dirichlet_process', 
                                             covariance_type='full')

    def fit(self, info_map):
        height, width = info_map.shape
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        data = np.column_stack((X.ravel(), Y.ravel(), info_map.ravel()))
        data = data[data[:, 2] > 0]  
        
        num_samples = min(5000, data.shape[0])  
        sampled_indices = np.random.choice(len(data), size=num_samples, p=data[:, 2] / np.sum(data[:, 2]))
        sampled_data = data[sampled_indices, :2]
        
        self.model.fit(sampled_data)
        
        gaussians = self.get_gaussians()
        
        if self.plot_gaussians:
            self.plot_results(info_map, gaussians, height, width)
        
        return gaussians

    def get_gaussians(self):
        gaussians = []
        for i in range(self.model.n_components):
            if self.model.weights_[i] > self.weight_threshold: 
                gaussians.append({
                    'weight': self.model.weights_[i],
                    'mean': self.model.means_[i],
                    'covariance': self.model.covariances_[i]
                })
        return gaussians

    def plot_results(self, info_map, gaussians, height, width):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(info_map, cmap='plasma', origin='lower')
        axes[0].set_title("Original Information Map")
        
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        xy = np.column_stack([X.ravel(), Y.ravel()])
        gaussian_map = np.zeros((height, width))

        for g in gaussians:
            mean = g['mean']
            cov = g['covariance']
            weight = g['weight']

            try:
                inv_cov = np.linalg.inv(cov)  
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov) 
            
            diff = xy - mean
            exponent = -0.5 * np.einsum('ij,ij->i', diff @ inv_cov, diff)
            gaussian_map += weight * np.exp(exponent).reshape(height, width)

        axes[1].imshow(gaussian_map, cmap='plasma', origin='lower')
        axes[1].set_title("DP-GMM")
        
        plt.show()

if __name__ == "__main__":
    generator = mapGenerator(100, 100, 5)
    info_map = generator.generate_map()
    
    dgmm = DirichletGMM(max_components=10, weight_threshold=1e-3, plot_gaussians=True)
    gaussians = dgmm.fit(info_map)
    
    for g in gaussians:
        print(f"Weight: {g['weight']}, Mean: {g['mean']}, Covariance: {g['covariance']}")
