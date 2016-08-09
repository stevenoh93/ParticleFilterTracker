"""Problem Set 7: Particle Filter Tracking"""

from ps7 import *

# I/O directories
input_dir = "input"
output_dir = "output"


# Driver/helper code
def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiate and run a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any keyword arguments.

    Parameters
    ----------
        pf_class: particle filter class to instantiate (e.g. ParticleFilter)
        video_filename: path to input video file
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int
        save_frames: dictionary of frames to save {<frame number>|'template': <filename>}
        kwargs: arbitrary keyword arguments passed on to particle filter class
    """

    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)  

            if False:  # For debugging, it displays every frame
                out_frame = frame.copy()
                pf.render(out_frame)
                cv2.imshow('Tracking', out_frame)
                cv2.waitKey(1)

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break


def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    # TODO: Implement ParticleFilter
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}  # suggested template window (dict)
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            84: os.path.join(output_dir, 'ps7-1-a-3.png'),
            144: os.path.join(output_dir, 'ps7-1-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=50, sigma_mse=None, sigma_dyn=None,
        template_coords=template_rect)  # TODO: specify other keyword args that your model expects.

    # TODO: Repeat 1a, but vary template window size and discuss trade-offs (no output images required)
    # TODO: Repeat 1a, but vary the sigma_MSE parameter (no output images required)
    # TODO: Repeat 1a, but try to optimize (minimize) num_particles (no output images required)

    # 1b
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}
    run_particle_filter(ParticleFilter,
        os.path.join(input_dir, "noisy_debate.mp4"),
        template_rect,
        {
            14: os.path.join(output_dir, 'ps7-1-b-1.png'),
            32: os.path.join(output_dir, 'ps7-1-b-2.png'),
            46: os.path.join(output_dir, 'ps7-1-b-3.png')
        },
        num_particles=50, sigma_mse=None, sigma_dyn=None, particle_sigma=None,
        template_coords=template_rect)  # TODO: specify other keyword args that your model expects.
    # 2a
    # TODO: Implement AppearanceModelPF (derived from ParticleFilter)
    # TODO: Run it on pres_debate.mp4 to track Romney's left hand, tweak parameters to track up to frame 140
    template_rect = {'x': None, 'y': None, 'w': None, 'h': None}  # TODO: Define the hand coordinate values
    run_particle_filter(AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "pres_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
            15: os.path.join(output_dir, 'ps7-2-a-2.png'),
            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
            140: os.path.join(output_dir, 'ps7-2-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=None, sigma_mse=None, sigma_dyn=None, particle_sigma=None, alpha=None,
        template_coords=template_rect)  # TODO: specify other keyword args that your model expects.

    # 2b
    # TODO: Run AppearanceModelPF on noisy_debate.mp4, tweak parameters to track hand up to frame 140
    template_rect = {'x': None, 'y': None, 'w': None, 'h': None}
    run_particle_filter(AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "noisy_debate.mp4"),  # input video
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
            15: os.path.join(output_dir, 'ps7-2-b-2.png'), #28
            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
            140: os.path.join(output_dir, 'ps7-2-b-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=None, sigma_mse=None, sigma_dyn=None, particle_sigma=None, alpha=None,
        template_coords=template_rect)  # TODO: specify other keyword args that your model expects.

    # 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF)
    # 3a
    template_rect = {'x': 320.8751, 'y': 175.1776, 'w': 103.5404, 'h': 129.0504}
    run_particle_filter(MeanShiftLitePF,
        os.path.join(input_dir, "pres_debate.mp4"),
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-3-a-1.png'),
            28: os.path.join(output_dir, 'ps7-3-a-2.png'),
            84: os.path.join(output_dir, 'ps7-3-a-3.png'),
            144: os.path.join(output_dir, 'ps7-3-a-4.png')
        },
        num_particles=None, sigma_mse=None, sigma_dyn=None, particle_sigma=None,
        chi_sigma=None, hist_bins_num=8, template_coords=template_rect)

    # 3b
    template_rect = {'x': None, 'y': None, 'w': None, 'h': None}
    run_particle_filter(MeanShiftLitePF,
        os.path.join(input_dir, "pres_debate.mp4"),
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-3-b-1.png'),
            15: os.path.join(output_dir, 'ps7-3-b-2.png'),
            50: os.path.join(output_dir, 'ps7-3-b-3.png'),
            140: os.path.join(output_dir, 'ps7-3-b-4.png')
        },
        num_particles=None, sigma_mse=None, sigma_dyn=None, particle_sigma=None,
        chi_sigma=None, hist_bins_num=None, template_coords=template_rect)

    # 4: Discussion problems. See problem set document.

    # 5: Implement a more sophisticated model to deal with occlusions and size/perspective changes
    template_rect = {'x': None, 'y': None, 'w': None, 'h': None}
    run_particle_filter(MDParticleFilter,
        os.path.join(input_dir, "pedestrians.mp4"),
        template_rect,
        {
            'template': os.path.join(output_dir, 'ps7-5-a-1.png'),
            40: os.path.join(output_dir, 'ps7-5-a-2.png'),
            100: os.path.join(output_dir, 'ps7-5-a-3.png'),
            240: os.path.join(output_dir, 'ps7-5-a-4.png')
        },
        num_particles=None, sigma_mse=None, sigma_dyn=None, particle_sigma=None, alpha=None,
        template_coords=template_rect)  # Add more if you need to

if __name__ == '__main__':
    main()