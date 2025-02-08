import numpy as np
from agent import Agent

import numpy as np
import matplotlib.pyplot as plt

import numpy as np

import numpy as np
import matplotlib.pyplot as plt

class GraphSubcomponent:
    def __init__(self, component_id, central_mean, central_cov, central_weight,
                 cw_mean, cw_cov, cw_weight, ccw_mean, ccw_cov, ccw_weight, grid_size=100):
        """
        Initializes a graph subcomponent with weighted spectral maps.

        Parameters:
        - component_id (int): ID of the component.
        - central_weight, cw_weight, ccw_weight (float): Importance weights (sum to 1).
        """
        self.component_id = component_id
        self.grid_size = grid_size

        # Store component weights
        self.central_weight = central_weight
        self.cw_weight = cw_weight
        self.ccw_weight = ccw_weight

        # Generate and normalize component maps
        self.central_map = self._generate_gaussian(central_mean, central_cov)
        self.cw_map = self._generate_gaussian(cw_mean, cw_cov)
        self.ccw_map = self._generate_gaussian(ccw_mean, ccw_cov)

        # Compute Fourier-transformed spectral maps, scaled by component weights
        self.central_spectral_map = self._compute_fourier(self.central_map) * self.central_weight
        self.cw_spectral_map = self._compute_fourier(self.cw_map) * self.cw_weight
        self.ccw_spectral_map = self._compute_fourier(self.ccw_map) * self.ccw_weight

        # Track agents assigned to each component
        self.drones = []
        self.quadrupeds = []

        # Initialize aggregated agent Fourier maps
        self.central_agent_map = np.zeros((grid_size, grid_size))
        self.cw_agent_map = np.zeros((grid_size, grid_size))
        self.ccw_agent_map = np.zeros((grid_size, grid_size))

        # Initialize sensing mismatch cost
        self.sensing_mismatch_cost = 0.0

    def _generate_gaussian(self, mean, cov):
        """Generates and normalizes a 100x100 Gaussian map."""
        x = np.linspace(-3, 3, self.grid_size)
        y = np.linspace(-3, 3, self.grid_size)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)
        norm_factor = 1.0 / (2 * np.pi * np.sqrt(det_cov))
        diff = pos - mean
        gaussian = norm_factor * np.exp(-0.5 * np.einsum('...i,ij,...j', diff, inv_cov, diff))

        return gaussian / np.sum(gaussian)  # Normalize total sum to 1

    def _compute_fourier(self, map_data):
        """Computes the magnitude spectrum of a 2D Fourier transform."""
        fourier_transform = np.fft.fft2(map_data)
        magnitude_spectrum = np.abs(np.fft.fftshift(fourier_transform))
        return magnitude_spectrum / np.sum(magnitude_spectrum)  # Normalize sum to 1

    def add_agent(self, agent, location):
        """
        Adds an agent and updates the respective Fourier-transformed agent map.
        """
        if agent._type == "drone":
            self.drones.append(agent)
        elif agent._type == "quadruped":
            self.quadrupeds.append(agent)

        if location == 'central':
            self.central_agent_map += self._compute_fourier(agent.sensing_map)
        elif location == 'cw':
            self.cw_agent_map += self._compute_fourier(agent.sensing_map)
        elif location == 'ccw':
            self.ccw_agent_map += self._compute_fourier(agent.sensing_map)
        else:
            raise ValueError("Location must be 'central', 'cw', or 'ccw'.")

        # Normalize agent maps
        if np.sum(self.central_agent_map) > 0:
            self.central_agent_map /= np.sum(self.central_agent_map)
        if np.sum(self.cw_agent_map) > 0:
            self.cw_agent_map /= np.sum(self.cw_agent_map)
        if np.sum(self.ccw_agent_map) > 0:
            self.ccw_agent_map /= np.sum(self.ccw_agent_map)

        # Update sensing mismatch cost
        self.compute_sensing_mismatch_cost()

    def compute_sensing_mismatch_cost(self):
        """
        Computes the total sensing-mismatch cost as the sum of MSE losses.
        """
        mse_central = np.mean((self.central_spectral_map - self.central_agent_map) ** 2)
        mse_cw = np.mean((self.cw_spectral_map - self.cw_agent_map) ** 2)
        mse_ccw = np.mean((self.ccw_spectral_map - self.ccw_agent_map) ** 2)

        self.sensing_mismatch_cost = mse_central + mse_cw + mse_ccw

    def plot_spectral_maps(self):
        """
        Plots spectral magnitude maps for each component and the corresponding agent subteams.
        """
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))

        axs[0, 0].imshow(self.central_spectral_map, cmap='viridis')
        axs[0, 0].set_title('Central Component Spectral Map')
        axs[0, 1].imshow(self.central_agent_map, cmap='inferno')
        axs[0, 1].set_title('Central Agent Spectral Map')

        axs[1, 0].imshow(self.cw_spectral_map, cmap='viridis')
        axs[1, 0].set_title('CW Component Spectral Map')
        axs[1, 1].imshow(self.cw_agent_map, cmap='inferno')
        axs[1, 1].set_title('CW Agent Spectral Map')

        axs[2, 0].imshow(self.ccw_spectral_map, cmap='viridis')
        axs[2, 0].set_title('CCW Component Spectral Map')
        axs[2, 1].imshow(self.ccw_agent_map, cmap='inferno')
        axs[2, 1].set_title('CCW Agent Spectral Map')

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Define Gaussian parameters for each component
    # Define Gaussian parameters and weights for each component
    central_mean, central_cov, central_weight = (0, 0), [[1, 0], [0, 1]], 0.4
    cw_mean, cw_cov, cw_weight = (1, 1), [[1, 0.2], [0.2, 1]], 0.35
    ccw_mean, ccw_cov, ccw_weight = (-1, -1), [[1, -0.2], [-0.2, 1]], 0.25

    # Create a GraphSubcomponent
    subcomponent = GraphSubcomponent(
        component_id=0,
        central_mean=central_mean, central_cov=central_cov, central_weight=central_weight,
        cw_mean=cw_mean, cw_cov=cw_cov, cw_weight=cw_weight,
        ccw_mean=ccw_mean, ccw_cov=ccw_cov, ccw_weight=ccw_weight
    )

    # Create drones and quadrupeds
    drone1 = Agent(variance=50, type="drone")
    quad1 = Agent(variance=20, type="quadruped")

    # Add them to the component
    subcomponent.add_agent(drone1, 'central')
    subcomponent.add_agent(quad1, 'cw')

    # Print sensing mismatch cost
    print("Sensing Mismatch Cost:", subcomponent.compute_sensing_mismatch_cost())

    # Plot spectral maps
    subcomponent.plot_spectral_maps()

