import napari
import numpy as np


pts_coordinates = np.random.random((500, 10, 3))  # N, t, xyz

viewer = napari.Viewer(ndisplay=3)
pts_layer = viewer.add_points(pts_coordinates, size=2)


napari.run()
