"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods."""

    def __init__(self, frame, template, **kwargs):
        """Initialize particle filter object.

        Parameters
        ----------
            frame: color BGR uint8 image of initial video frame, values in [0, 255]
            template: color BGR uint8 image of patch to track, values in [0, 255]
            kwargs: keyword arguments needed by particle filter model, including:
            - num_particles: number of particles
            - sigma_mse: sigma value used in the similarity measure
            - sigma_dyn: sigma value that can be used when adding gaussian noise to u and v
        """
        self.num_particles = kwargs.get('num_particles', 100)  # required by the autograder
        self.sigma_mse = kwargs.get('sigma_mse', None)  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn', None)  # required by the autograder
        self.template_rect = kwargs.get('template_coords', None)
        self.particles = None  # Here you will store your particles. You should initialize them in this step.
                               #  required by the autograder
        self.weights = None  # List / 1D array of weights for each particle. Hint: initialize them with a uniform
                             # normalized distribution (equal weight for each one). Required by the autograder
        self.template = template
        self.frame = frame
        # TODO: Your code here - extract any additional keyword arguments you need and initialize state

    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        pass  # TODO: Your code here - use the frame as a new observation (measurement) and update model

    def render(self, frame_out):
        """Visualize current particle filter state.

        Parameters
        ----------
            frame_out: copy of frame to overlay visualization on
        """
        # Note: This may not be called for all frames, so don't do any model updates here!
        # These steps will calculate the weighted mean. The resulting values should represent the
        # tracking window center point.
        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i][1] * self.weights[i]
            v_weighted_mean += self.particles[i][0] * self.weights[i]

        # Complete the rest of the code as instructed.

        pass  # TODO: Your code here - draw particles, tracking window and a circle to indicate spread


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        self.alpha = kwargs.get('alpha', None)  # required by the autograder
        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update

    # TODO: Override render() if desired (shouldn't have to, ideally)


class MeanShiftLitePF(ParticleFilter):
    """A variation of particle filter tracker that uses the color distribution of the patch."""

    def __init__(self, frame, template, **kwargs):
        """Initialize Mean Shift Lite particle filter object (parameters same as ParticleFilter)."""
        self.chi_sigma = kwargs.get('chi_sigma', None)  # required by the autograder
        self.num_bins = kwargs.get('hist_bins_num', 8)  # required by the autograder
        super(MeanShiftLitePF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update

    # TODO: Override render() if desired (shouldn't have to, ideally)


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initialize MD particle filter object (parameters same as ParticleFilter)."""
        self.alpha = kwargs.get('alpha', None)
        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update

    # TODO: Override render() if desired (shouldn't have to, ideally)
