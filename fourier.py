import numpy as np
import matplotlib.pyplot as plt
from mapGenerator import mapGenerator

class FourierMagnitude:
    def __init__(self, map_data, plot_results=True):
        """
        Initializes the Fourier magnitude class.
        :param map_data: 2D numpy array representing the input map.
        :param plot_results: Boolean flag to enable visualization (default: True).
        """
        self.map_data = map_data
        self.plot_results = plot_results
        self.magnitude_spectrum = None

    def compute_magnitude(self):
        """
        Computes the magnitude spectrum of the 2D Fourier Transform.
        """
        fft_transform = np.fft.fft2(self.map_data)
        fft_transform = np.fft.fftshift(fft_transform)  # Shift zero frequency to the center
        self.magnitude_spectrum = np.abs(fft_transform)

        if self.plot_results:
            self.plot_results_figure()

        return self.magnitude_spectrum

    def plot_results_figure(self):
        """
        Plots both the original map and the magnitude spectrum in log scale.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot original information map
        axes[0].imshow(self.map_data, cmap='viridis', origin='lower')
        axes[0].set_title("Original Information Map")
        axes[0].set_xlabel("X Coordinate")
        axes[0].set_ylabel("Y Coordinate")

        # Compute log-scaled magnitude spectrum
        magnitude_log = np.log1p(self.magnitude_spectrum)  # log(1 + magnitude) avoids log(0) issues
        
        # Plot magnitude spectrum
        im = axes[1].imshow(magnitude_log, cmap='inferno', origin='lower')
        axes[1].set_title("Magnitude Spectrum (Log Scale)")
        axes[1].set_xlabel("X Frequency")
        axes[1].set_ylabel("Y Frequency")
        fig.colorbar(im, ax=axes[1], label="Log Magnitude")

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    generator = mapGenerator(100, 100, 5)
    info_map = generator.generate_map()

    fourier_magnitude = FourierMagnitude(info_map, plot_results=True)
    magnitude_spectrum = fourier_magnitude.compute_magnitude()
    
    print("Magnitude Spectrum Shape:", magnitude_spectrum.shape)
