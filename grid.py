import train
import misc
import os
import config
import random
import numpy as np
import math
import pickle

def create_image_grid(images, grid_size=None, padding=5):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) / grid_w + 1, 1)
    grid = np.ones(list(images.shape[1:-2]) + [grid_h * img_h + padding * (grid_h - 1), grid_w * img_w + padding * (grid_w - 1)], dtype=images.dtype) * 255
    x = 0
    y = 0
    grid[..., y : y + img_h, x : x + img_w] = images[0]
    for idx in xrange(1, num):
        i = idx % grid_w
        j = idx / grid_w
        x = i * (img_w + padding)
        y = j * (img_h + padding)
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def slerp(val, low, high):
     omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
     so = np.sin(omega)
     if so == 0:
         return (1.0-val) * low + val * high # L'Hopital's rule/LERP
     return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def get_2d_slerp_points(num_points, plane):
    points = []
    for i in np.arange(0.0, 1.0, 1.0 / math.sqrt(num_points)):
       x = slerp(i, plane[0], plane[1])
       for j in np.arange(0.0, 1.0, 1.0 / math.sqrt(num_points)):
           y = slerp(j, x, plane[2])
           points = np.concatenate((points, y), axis=0)
    return points.reshape((num_points, 512)).astype("float32")

net = train.imgapi_load_net("000-wikiart-512x512", "/home/cameron/Projects/progressive_growing_of_gans/results/000-wikiart-512x512/network-snapshot-006200.pkl", load_dataset=False, random_seed=np.int64(10000*random.random()))
for i in range(100):
    plane = train.random_latents(3, net.G.input_shape)
    grid_shape = 10 + int(random.random() * 20)
    points = get_2d_slerp_points(grid_shape * grid_shape, plane)
    images = net.gen_fn(points, net.example_labels[:512])
    grid = create_image_grid(images, (grid_shape, grid_shape), padding=10)
    filename = os.path.join(config.result_dir, "grids", "grid%s.%s")
    file_num = 1
    while os.path.exists(filename % (file_num, "png")):
      file_num += 1

    misc.save_image(grid, filename % (file_num, "png"), drange=[0,255])
    np.save(filename % (file_num, "npy"), points)
