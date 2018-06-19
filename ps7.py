"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

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
                    - sigma_exp (float): sigma value used in the similarity measure.
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
        self.particles = np.random.rand(self.num_particles, 2) * [frame.shape[0] - (int)(self.template_rect['h']), frame.shape[1] - (int)(self.template_rect['w'])]
        self.particles += [(int)(self.template_rect['h']/2), (int)(self.template_rect['w']/2)]
        # rows = np.linspace(self.template_rect['h']/2, frame.shape[0] - (int)(self.template_rect['h'])/2, num=self.num_particles)
        # cols = np.linspace(self.template_rect['w']/2, frame.shape[1] - (int)(self.template_rect['w'])/2, num=self.num_particles)
        # self.particles = np.hstack((np.reshape(rows,(self.num_particles,1)), np.reshape(cols,(self.num_particles,1))))
        self.weights = np.ones(self.num_particles, dtype='float') / self.num_particles
        self.org_count = self.num_particles
        self.count = 0
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
        nu = 0
        frame_cp = np.copy(frame)
        frame_gray = 0.3*frame_cp[:,:,2] + 0.58*frame_cp[:,:,1] + 0.12*frame_cp[:,:,0]
        templ = 0.3*self.template[:,:,2] + 0.58*self.template[:,:,1] + 0.12*self.template[:,:,0]
        if self.count == 0:
            self.num_particles = 1000
            if self.weights.shape[0] != 1000:
                self.weights = np.ones(self.num_particles, dtype='float') / self.num_particles
                self.particles = np.random.rand(self.num_particles, 2) * [frame.shape[0] - (int)(self.template_rect['h']), frame.shape[1] - (int)(self.template_rect['w'])]
                self.particles += [(int)(self.template_rect['h']/2), (int)(self.template_rect['w']/2)]
        elif self.count == 10:
            self.num_particles = self.org_count
            self.weights = np.sort(self.weights)[-self.num_particles:]
            self.weights /= np.sum(self.weights)
        new_weights = np.array([])
        new_particles = np.array([])
        m, n, c = self.template.shape
        m+=0.0
        n+=0.0
        fs = frame_gray.shape
        num_part = 0
        # for i in range(0,self.num_particles):
        while num_part < self.num_particles:
            # idx = self.findNextIdx(sample_idxs)
            idx = np.random.choice(self.num_particles, 1, p=self.weights)[0]
            new_pos = self.particles[idx, :]
            new_pos += np.random.normal(0, self.sigma_dyn, size=new_pos.shape)
            # if new_pos[0]-m/2 < 0 or fs[0]-new_pos[0]-m/2 <= 0 or new_pos[1]-n/2 < 0 or fs[1]-new_pos[1]-n/2 <= 0:
            #     continue
            if new_pos[0]-m/2 < 0:
                new_pos[0] = m/2
            if fs[0]-new_pos[0]-m/2 <= 0:
                new_pos[0] = fs[0]-m/2-1
            if new_pos[1]-n/2 < 0:
                new_pos[1] = n/2
            if fs[1]-new_pos[1]-n/2 < 0:
                new_pos[1] = fs[1]-n/2-1
            diff = templ - frame_gray[(int)(new_pos[0]-m/2):(int)(new_pos[0]+m/2), (int)(new_pos[1]-n/2):(int)(new_pos[1]+n/2)]
            diffsq = diff*diff
            mse = np.mean(diffsq)
            new_weights = np.append(new_weights, np.exp(-mse/(2*self.sigma_exp*self.sigma_exp)))
            new_particles = np.append(new_particles, [(int)(new_pos[0]), (int)(new_pos[1])])
            num_part += 1
            nu += new_weights[num_part-1]
        pass
        if nu == 0:
            self.weights = np.ones(self.num_particles, dtype='float') / new_weights.shape[0]
        else:
            self.weights = new_weights/nu
        self.particles = np.reshape(new_particles, (num_part, 2))
        self.count += 1
        # self.render(frame)

    def findNextIdx(self, idxs):
        idx = np.nonzero(idxs)[0]
        if idx.shape[0] == 0:
            return -1
        ret = idx[0]
        idxs[ret] -= 1
        return ret

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
        dist = 0

        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, ((int)(self.particles[i,1]),(int)(self.particles[i,0])), 2, (0,0,255), thickness=1)
        for i in range(self.num_particles):
            dist += np.linalg.norm([u_weighted_mean-self.particles[i,0], v_weighted_mean-self.particles[i,1]]) * self.weights[i]
        tl = ((int)(v_weighted_mean - self.template.shape[1]/2), (int)(u_weighted_mean - self.template.shape[0]/2))
        br = ((int)(v_weighted_mean + self.template.shape[1]/2), (int)(u_weighted_mean + self.template.shape[0]/2))
        cv2.rectangle(frame_in,tl,br,(0,255,0),2)
        cv2.circle(frame_in, ((int)(v_weighted_mean),(int)(u_weighted_mean)), (int)(dist), (255,0,0), thickness=2)
        # cv2.imshow('image', frame_in)
        # cv2.waitKey(1)
        # g = cv2.cvtColor(self.template.astype('float32'),  cv2.COLOR_BGR2GRAY)
        # tmplgray = cv2.normalize(g, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # cv2.imshow('templ', tmplgray)
        # cv2.waitKey(1)


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
        self.count = 0
        # self.fig=plt.figure()
        # self.window = self.fig.add_subplot(111)
        # self.img = self.window.imshow(self.template)
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        nu = 0
        frame_cp = np.copy(frame)
        # frame_gray = cv2.cvtColor(frame_cp, cv2.COLOR_RGB2GRAY).astype('float')
        frame_gray = 0.3*frame_cp[:,:,2] + 0.58*frame_cp[:,:,1] + 0.12*frame_cp[:,:,0]
        templ = 0.3*self.template[:,:,2] + 0.58*self.template[:,:,1] + 0.12*self.template[:,:,0]
        new_weights = np.array([])
        new_particles = np.array([])
        m, n, c = self.template.shape
        m+=0.0
        n+=0.0
        fs = frame_gray.shape
        num_part = 0
        # for i in range(0,self.num_particles):
        while num_part < self.num_particles:
            # idx = self.findNextIdx(sample_idxs)
            idx = np.random.choice(self.num_particles, 1, p=self.weights)[0]
            new_pos = self.particles[idx, :]
            new_pos += np.random.normal(0, self.sigma_dyn, size=new_pos.shape)
            if new_pos[0]-m/2 < 0:
                new_pos[0] = m/2
            if fs[0]-new_pos[0]-m/2 <= 0:
                new_pos[0] = fs[0]-m/2-1
            if new_pos[1]-n/2 < 0:
                new_pos[1] = n/2
            if fs[1]-new_pos[1]-n/2 < 0:
                new_pos[1] = fs[1]-n/2-1
            diff = templ - frame_gray[(int)(new_pos[0]-m/2):(int)(new_pos[0]+m/2), (int)(new_pos[1]-n/2):(int)(new_pos[1]+n/2)]
            diffsq = diff*diff
            mse = np.mean(diffsq)
            new_weights = np.append(new_weights, np.exp(-mse/(2*self.sigma_exp*self.sigma_exp)))
            new_particles = np.append(new_particles, [(int)(new_pos[0]), (int)(new_pos[1])])
            num_part += 1
            nu += new_weights[num_part-1]
        pass
        if nu == 0:
            self.weights = np.ones(self.num_particles, dtype='float') / new_weights.shape[0]
        else:
            self.weights = new_weights/nu
        self.particles = np.reshape(new_particles, (num_part, 2))
        u_weighted_mean=0.0
        v_weighted_mean=0.0
        for i in range(self.num_particles):
            u_weighted_mean += self.particles[i, 0] * self.weights[i]
            v_weighted_mean += self.particles[i, 1] * self.weights[i]
        new_templ = frame[(int)(u_weighted_mean-m/2):(int)(u_weighted_mean+m/2), (int)(v_weighted_mean-n/2):(int)(v_weighted_mean+n/2), :]
        self.template = self.alpha * new_templ + (1-self.alpha)*self.template
        self.count += 1
        # if self.count == 1:
        #     cv2.imwrite('out0.jpg', self.template)
        # if self.count == 2:
        #     cv2.imwrite('out01.jpg', self.template)
        # if self.count == 3:
        #     cv2.imwrite('out02.jpg', self.template)
        # if self.count == 4:
        #     cv2.imwrite('out03.jpg', self.template)
        # if self.count == 5:
        #     cv2.imwrite('out04.jpg', self.template)
        # if self.count == 10:
        #     cv2.imwrite('out1.jpg', self.template)
        # if self.count == 20:
        #     cv2.imwrite('out2.jpg', self.template)
        # if self.count == 30:
        #     cv2.imwrite('out3.jpg', self.template)
        # if self.count == 40:
        #     cv2.imwrite('out4.jpg', self.template)
        # if self.count == 50:
        #     cv2.imwrite('out5.jpg', self.template)
        # if self.count == 60:
        #     cv2.imwrite('out6.jpg', self.template)
        # if self.count == 70:
        #     cv2.imwrite('out7.jpg', self.template)

        # self.render(frame)


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
        nu = 0
        frame_cp = np.copy(frame)
        templ = self.template
        totalTmplHist = self.findHist(templ)
        new_weights = np.array([])
        new_particles = np.array([])
        m, n, c = self.template.shape
        m+=0.0
        n+=0.0
        fs = frame_cp.shape
        num_part = 0
        # for i in range(0,self.num_particles):
        while num_part < self.num_particles:
            # idx = self.findNextIdx(sample_idxs)
            idx = np.random.choice(self.num_particles, 1, p=self.weights)[0]
            new_pos = self.particles[idx, :]
            new_pos += np.random.normal(0, self.sigma_dyn, size=new_pos.shape)
            if new_pos[0]-m/2 < 0:
                new_pos[0] = m/2
            if fs[0]-new_pos[0]-m/2 <= 0:
                new_pos[0] = fs[0]-m/2-1
            if new_pos[1]-n/2 < 0:
                new_pos[1] = n/2
            if fs[1]-new_pos[1]-n/2 < 0:
                new_pos[1] = fs[1]-n/2-1
            totalPartHist = self.findHist(frame_cp[(int)(new_pos[0]-m/2):(int)(new_pos[0]+m/2), (int)(new_pos[1]-n/2):(int)(new_pos[1]+n/2),:])
            diff = totalTmplHist - totalPartHist
            diffsq = diff * diff
            sums = totalTmplHist + totalPartHist
            sums[np.where(sums==0)] += 100000
            div = diffsq / sums
            chi = 0.5 * np.sum(div)
            new_weights = np.append(new_weights, np.exp(-chi/(2*self.sigma_exp*self.sigma_exp)))
            new_particles = np.append(new_particles, [(int)(new_pos[0]), (int)(new_pos[1])])
            num_part += 1
            nu += new_weights[num_part-1]
        pass
        if nu == 0:
            self.weights = np.ones(self.num_particles, dtype='float') / new_weights.shape[0]
        else:
            self.weights = new_weights/nu
        self.particles = np.reshape(new_particles, (num_part, 2))
        # self.render(frame)

    def findHist(self, image):
        btmplHist = cv2.calcHist([image],[0], None, [self.num_bins], [0, 256])
        gtmplHist = cv2.calcHist([image],[1], None, [self.num_bins], [0, 256])
        rtmplHist = cv2.calcHist([image],[2], None, [self.num_bins], [0, 256])
        return np.append(np.append(btmplHist, gtmplHist), rtmplHist)

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
