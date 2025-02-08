import numpy as np
import matplotlib.pyplot as plt

class SpectralMismatchCost:
    def __init__(self, info_map, agent_spectral_abilities):
        """
        Initialize the cost computation with the given information map and agent sensing spectral abilities.
        
        :param info_map: 2D numpy array representing the information distribution.
        :param agent_spectral_abilities: List of 2D numpy arrays, each representing an agent's spectral sensing ability.
        """
        
        self.info_map = info_map
        self.agent_spectral_abilities = agent_spectral_abilities
        
        # Compute the Fourier coefficients of the information map
        self.info_spectrum = np.fft.fft2(info_map)
        self.info_spectrum_shifted = np.fft.fftshift(self.info_spectrum)
        
        # Compute weights for low-frequency emphasis
        self.weights = self._compute_spectral_weights(info_map.shape)
    
    def _compute_spectral_weights(self, shape):
        """
        Compute spectral weights to emphasize low-frequency components.
        """
        nx, ny = shape
        kx, ky = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny), indexing="ij")
        frequency_magnitude = np.sqrt(kx**2 + ky**2)
        weights = (1 + frequency_magnitude**2) ** -((2+1)/2)  # d=2 in our case
        return np.fft.fftshift(weights)  # Shift to align with FFT output
    
    def compute_mismatch_cost(self):
        """
        Compute the spectral mismatch cost based on the ergodic spectral metric.
        """
        total_agent_spectrum = np.zeros_like(self.info_spectrum, dtype=np.complex128)
        
        # Sum up the Fourier coefficients of all agents' spectral abilities
        for agent_spectral in self.agent_spectral_abilities:
            agent_spectrum = np.fft.fft2(agent_spectral)
            agent_spectrum_shifted = np.fft.fftshift(agent_spectrum)
            total_agent_spectrum += agent_spectrum_shifted  # Combine agent capabilities
        
        # Compute the spectral mismatch cost
        spectral_difference = np.abs(total_agent_spectrum - self.info_spectrum_shifted) ** 2
        weighted_difference = self.weights * spectral_difference
        cost = np.sum(weighted_difference)
        
        return cost

# Example Usage
def generate_information_map(size=100, centers=[(30, 30), (70, 70)], sigma=10):
    """Generate a 2D Gaussian information distribution."""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    info_map = np.zeros((size, size))
    for center in centers:
        info_map += np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2 * sigma**2))
    return info_map

def generate_agent_spectral_ability(size=100, freq_scale=5):
    """Generate an agent's spectral ability function, with better sensing at low frequencies."""
    kx, ky = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    spectrum = np.exp(-freq_scale * (kx**2 + ky**2))  # Gaussian decay in frequency domain
    return np.fft.ifft2(np.fft.ifftshift(spectrum)).real  # Transform back to spatial domain

# Generate example data
size = 100
info_map = generate_information_map(size=size)

# Create three agents with different spectral sensing capabilities
agents_spectral = [generate_agent_spectral_ability(size=size, freq_scale=scale) for scale in [3, 5, 8]]

# Compute the mismatch cost
cost_computer = SpectralMismatchCost(info_map, agents_spectral)
mismatch_cost = cost_computer.compute_mismatch_cost()

print(f"Spectral Mismatch Cost: {mismatch_cost:.4f}")

# Visualization
fig, ax = plt.subplots(1, len(agents_spectral) + 1, figsize=(12, 4))
ax[0].imshow(info_map, cmap="viridis")
ax[0].set_title("Information Map")

for i, agent_spectral in enumerate(agents_spectral):
    ax[i+1].imshow(agent_spectral, cmap="inferno")
    ax[i+1].set_title(f"Agent {i+1} Spectral Ability")

plt.show()
