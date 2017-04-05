"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods. Refer to the method
    run_particle_filter( ) in experiment.py to understand how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles. This should be a N x 2 array where
                                        N = self.num_particles. This component is used by the autograder so make sure
                                        you define it appropriately.
        - self.weights (numpy.array): Array of N weights, one for each particle.
                                      Hint: initialize them with a uniform normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video frame that will be used as the template to
                                       track.
        - self.frame (numpy.array): Current video frame from cv2.VideoCapture().

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame, values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track, values in [0, 255].
            kwargs: keyword arguments needed by particle filter model, including:
                    - num_particles (int): number of particles.
                    - sigma_mse (float): sigma value used in the similarity measure.
                    - sigma_dyn (float): sigma value that can be used when adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y, width, and height values.
        """

        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        self.particles = None  # Todo: Initialize your particles array. Read the docstring.
        self.weights = None  # Todo: Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.

        """

        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """

        return self.weights

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None (do not include a return call). This function
        should update the particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the image. This means you should address
        particles that are close to the image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        pass

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model updates here!
        These steps will calculate the weighted mean. The resulting values should represent the
        tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay each successive
        frame with the following elements:

        - Every particle's (u, v) location in the distribution should be plotted by drawing a
          colored dot point on the image. Remember that this should be the center of the window,
          not the corner.
        - Draw the rectangle of the tracking window associated with the Bayesian estimate for
          the current location which is simply the weighted mean of the (u, v) of the particles.
        - Finally we need to get some sense of the standard deviation or spread of the distribution.
          First, find the distance of every particle to the weighted mean. Next, take the weighted
          sum of these distances and plot a circle centered at the weighted mean with this radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the particle filter.
        """

        u_weighted_mean = 0
        v_weighted_mean = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        pass


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter object (parameters are the same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above. There is one element that is added
        called alpha which is explained in the problem set documentation. By calling super(...) all the elements used
        in ParticleFilter will be inherited so you do not have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        pass


class MeanShiftLitePF(ParticleFilter):
    """A variation of particle filter tracker that uses the color distribution of the patch."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the Mean Shift Lite particle filter object (parameters are the same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above. There is one element that is added
        called alpha which is explained in the problem set documentation. By calling super(...) all the elements used
        in ParticleFilter will be inherited so you don't have to declare them again."""

        super(MeanShiftLitePF, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.num_bins = kwargs.get('hist_bins_num', 8)  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "Mean Shift Lite" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        pass


class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object (parameters same as ParticleFilter).

        The documentation for this class is the same as the ParticleFilter above.
        By calling super(...) all the elements used in ParticleFilter will be inherited so you
        don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your implementation, you may comment out this
        function and use helper methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        pass
