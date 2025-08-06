from scene.gaussian_model import GaussianModel

class GaussianSegment(GaussianModel):
    
    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, 
                 force_sh_3d: bool = False, sh_degree_t : int = 0, level : int = 0, start_time : float = 0, end_time : float = 0):
        super().__init__(sh_degree, gaussian_dim, time_duration, rot_4d, force_sh_3d, sh_degree_t)
        self.level = level
        self.start_time = start_time
        self.end_time = end_time

    
