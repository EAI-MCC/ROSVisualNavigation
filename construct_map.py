import numpy as np 
from argparse import Namespace 
import matplotlib.pyplot as plt 

ANGLE_EPS = 0.001

class Map(object):
    def __init__(self, width, height, fov=90):
        self.camera_matrix = self._get_camera_matrix(width, height, fov)
        self.vision_range = 320
        self.map_size_cm = (11100, 11100) # (783, 783)*2 # dm=10cn

        self.resolution = 5 # 1 pixel per cm
        self.z_bins = [50, 150] # 25
        self.du_scale = 2
        self.visualize = 1
        self.obs_threshold = 0.2
        self.map = np.zeros((self.map_size_cm[0] // self.resolution, 
                             self.map_size_cm[1] // self.resolution, len(self.z_bins)+1), dtype=np.float32)
        self.human_map = np.zeros((self.map_size_cm[0] // self.resolution,
                                   self.map_size_cm[1] // self.resolution, len(self.z_bins)+1), dtype=np.float32)
        self.agent_height = 92
        self.agent_view_angle = 0

        self.map_size = self.map_size_cm[0] // self.resolution 
    
    def _get_camera_matrix(self, width, height, fov):
        xc = (width - 1.) / 2.
        zc = (height - 1.) / 2.
        f = (width / 2.) / np.tan(np.deg2rad(fov/2.))

        camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
        camera_matrix = Namespace(**camera_matrix)
        return camera_matrix
    
    def _preprocess_depth(self, depth):
        depth = depth[:, :, 0] * 1.
        mask2 = depth > 0.99
        depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0] = depth[:, i].max()
        
        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth * 1000.
        return depth

    def _get_cloud_from_z(self, Y):
        """
        Project the depth image Y into a 3D point cloud.

        Parameters:
        ------------------------
        depth : ...xHxW

        Returns:
        ------------------------
        X: positive going right.
        Y: positive into the image.
        Z: positive up in the image.
        XYZ: ...xHxWx3
        """
        x, z = np.meshgrid(np.arange(Y.shape[-1]), np.arange(Y.shape[-2]-1, -1, -1))
        for i in range(Y.ndim - 2):
            x = np.expand_dims(x, axis=0)
            z = np.expand_dims(z, axis=0)
        
        X = (x[::self.du_scale, ::self.du_scale] - self.camera_matrix.xc)*Y[::self.du_scale, ::self.du_scale] / self.camera_matrix.f 
        Z = (z[::self.du_scale, ::self.du_scale] - self.camera_matrix.zc)*Y[::self.du_scale, ::self.du_scale] / self.camera_matrix.f 
        XYZ = np.concatenate((X[..., np.newaxis], Y[::self.du_scale, ::self.du_scale][..., np.newaxis],
                              Z[..., np.newaxis]), axis=X.ndim)
        return XYZ 
    
    def _get_r_matrix(self, ax_, angle):
        ax = ax_ / np.linalg.norm(ax_)

        if np.abs(angle) > ANGLE_EPS:
            S_hat = np.array([[0., -ax[2], ax[1]],
                              [ax[2], 0., -ax[0]],
                              [-ax[1], ax[0], 0.]], dtype=np.float32)
            ## Rodrigue's Formula
            R = np.eye(3) + np.sin(angle) * S_hat + (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
        else:
            R = np.eye(3)
        return R 
    
    
    def _transform_camera_view(self, XYZ):
        """
        Transform the point cloud into geocentric frame to account for camera elevation and angle.

        Parameters:
        ----------------------------
        XYZ: ...x3

        Returns:
        ----------------------------
        XYZ: ...x3
        """
        R = self._get_r_matrix([1., 0., 0.], angle=np.deg2rad(self.agent_view_angle))
        XYZ = np.matmul(XYZ.reshape(-1,3), R.T).reshape(XYZ.shape)
        XYZ[...,2] = XYZ[...,2] + self.agent_height
        return XYZ 
    
    def _transform_pose(self, XYZ, current_pose):
        """
        Transform the point cloud into geocentric frame to account for camera position.
        
        Parameters
        ----------------------
        XYZ: ...x3
        current_pose: camera position (x, y, theta(radians)).
        
        Returns
        ----------------------
        XYZ: ...x3.
        """
        R = self._get_r_matrix([1., 0., 0.], angle=current_pose[2] - np.pi / 2.)
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
        XYZ[...,0] = XYZ[...,0] + current_pose[0]
        XYZ[...,1] = XYZ[...,1] + current_pose[1]
        return XYZ
    
    def _bin_points(self, XYZ_cms, map_size, z_bins, xy_resolution):
        """
        Bins points into xy-z bins
        
        Parameters
        ----------------------
        XYZ_cms: ...xHxWx3
        
        Returns
        ----------------------
        counts: ... x map_size x map_size x (len(z_bins) + 1).
        """

        sh = XYZ_cms.shape
        XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
        n_z_bins = len(z_bins) + 1

        counts = []
        for XYZ_cm in XYZ_cms:
            isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
            X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
            Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
            Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)

            isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                                Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
            isvalid = np.all(isvalid, axis=0)

            ind = (Y_bin*map_size + X_bin)*n_z_bins + Z_bin
            ind[np.logical_not(isvalid)] = 0

            count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                                minlength=map_size*map_size*n_z_bins)
            counts = np.reshape(count, [map_size, map_size, n_z_bins])

        counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

        return counts 
    

    def update_map(self, obs, depth, detected_region, current_pose, flag=1):
        pose = current_pose.copy()

        # to be modified
        pose[0] %= self.map_size_cm[0]
        pose[1] %= self.map_size_cm[1]
        pose[2] = np.deg2rad(pose[2])

        depth = np.expand_dims(depth.copy(), axis=2)
       # depth = self._preprocess_depth(depth)
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range*self.resolution] = np.NaN
            print("nan")

        point_cloud = self._get_cloud_from_z(depth)

        agent_view = self._transform_camera_view(point_cloud)

        shift_loc = [self.vision_range*self.resolution//2, 0, np.pi/2.]

        agent_view_centered = self._transform_pose(agent_view, shift_loc)

        agent_view_flat = self._bin_points(agent_view_centered, self.vision_range, self.z_bins, self.resolution)

        agent_view_cropped = agent_view_flat[:, :, 1]
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.

        geocentric_pc = self._transform_pose(agent_view, pose)
        geocentric_flat = self._bin_points(geocentric_pc, self.map.shape[0],
                                           self.z_bins, self.resolution)
        
        if flag == 1:
            self.map += geocentric_flat

            map_gt = self.map[:, :, 1] / self.obs_threshold
            map_gt[map_gt >= 0.5] = 1.
            map_gt[map_gt < 0.5] = 0.

            explored_gt = self.map.sum(2)
            explored_gt[explored_gt > 1] = 1.
        else:
            self.human_map += geocentric_flat 

            map_gt = self.human_map[:, :, 1] / self.obs_threshold
            map_gt[map_gt >= 0.5] = 1.
            map_gt[map_gt < 0.5] = 0.

            explored_gt = self.human_map.sum(2)
            explored_gt[explored_gt > 1] = 1.
        
        return agent_view_cropped, map_gt, agent_view_explored, explored_gt, depth, point_cloud


    def reset_map(self):
        self.map = np.zeros((self.map_size_cm[0] // self.resolution, self.map_size_cm[1]//self.resolution,
                             len(self.z_bins)+1), dtype=np.float32)
    
    def reset_human_map(self):
        self.human_map = np.zeros((self.map_size_cm[0] // self.resolution, self.map_size_cm[1] // self.resolution,
                                   len(self.z_bins)+1), dtype=np.float32)
    
    def vis_pointcloud(self, point_cloud1, point_cloud2, iter):
        point_cloud1 = np.reshape(point_cloud1, (-1, 3))
        point_cloud2 = np.reshape(point_cloud2, (-1, 3))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(point_cloud1[:, 0], point_cloud1[:, 1], point_cloud1[:, 2], c='b', marker='x')
        ax.scatter3D(point_cloud2[:, 0], point_cloud2[:, 1], point_cloud2[:, 2], c='r', marker='x')
        ax.axis()
        
        plt.savefig("results/point_clouds/point_clouds_%d.png" % iter)
        plt.close("all")
