import numpy as np

class mapGenerator:
    def __init__(self, width, height, num_peaks):
        self.width = width
        self.height = height
        self.num_peaks = num_peaks
        
    def generate_map(self):
        info_map = np.zeros((self.height, self.width))
        
        for _ in range(self.num_peaks):
            x_peak = np.random.randint(0, self.width)
            y_peak = np.random.randint(0, self.height)
            std_x = np.random.uniform(self.width * 0.05, self.width * 0.2)  
            std_y = np.random.uniform(self.height * 0.05, self.height * 0.2)  
            peak_density = np.random.uniform(0.5, 5.0)  
            
            x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
            gaussian_peak = peak_density * np.exp(-(((x - x_peak) ** 2) / (2 * std_x ** 2) + 
                                                    ((y - y_peak) ** 2) / (2 * std_y ** 2)))
            info_map += gaussian_peak
        
        return info_map

# Example usage
if __name__ == "__main__":
    generator = mapGenerator(100, 100, 5)
    info_map = generator.generate_map()
    import matplotlib.pyplot as plt
    plt.imshow(info_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
