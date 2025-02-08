import numpy as np

class Agent:
    def __init__(self, variance, type, map_size=100):
        self.map_size = map_size
        self.variance = variance
        self.sensing_map = self._generate_gaussian_map()

        self.current_component = None

        self._prob_cw = 0.0
        self._prob_ccw = 0.0

        self._cw_component_id = None
        self._ccw_component_id = None

        self._type = type

    def _generate_gaussian_map(self):
        """Creates a 2D Gaussian centered in the grid."""
        x = np.linspace(-1, 1, self.map_size)
        y = np.linspace(-1, 1, self.map_size)
        X, Y = np.meshgrid(x, y)
        gaussian = np.exp(-((X**2 + Y**2) / (2 * self.variance**2)))
        return gaussian

    def set_transition_probabilities(self, prob_cw, prob_ccw):
        """Sets the transition probabilities to the CW and CCW neighboring components."""
        if prob_cw + prob_ccw > 1.0:
            raise ValueError("Transition probabilities cannot sum to more than 1.")
        self._prob_cw = prob_cw
        self._prob_ccw = prob_ccw

    def set_type(self, type_id):
        self._type = type_id

    def set_adjacent_components(self, cw_component_id, ccw_component_id):
        """Sets the IDs of the CW and CCW neighboring components."""
        self._cw_component_id = cw_component_id
        self._ccw_component_id = ccw_component_id

    def sample_transition(self):
        """Samples a transition based on the probabilities and updates the current component."""
        rand_val = np.random.rand()
        if rand_val < self._prob_cw:
            self.current_component = self._cw_component_id
        elif rand_val < self._prob_cw + self._prob_ccw:
            self.current_component = self._ccw_component_id

if __name__ == "__main__":
    agent = Agent(variance=5.0)
    agent.set_transition_probabilities(0.3, 0.4)
    agent.set_adjacent_components(1, 2)

    print("Initial Component:", agent.current_component)
    agent.sample_transition()
    print("New Component after transition:", agent.current_component)
