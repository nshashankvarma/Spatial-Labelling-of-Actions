import cv2
import numpy as np
import matplotlib as plt

import skimage.draw as draw

# Calculating row and column gradients
def grad(channel):
    g_row = np.empty(channel.shape, dtype=channel.dtype)
    g_row[0, :] = 0
    g_row[-1, :] = 0
    g_row[1:-1, :] = channel[2:, :] - channel[:-2, :]
    g_col = np.empty(channel.shape, dtype=channel.dtype)
    g_col[:, 0] = 0
    g_col[:, -1] = 0
    g_col[:, 1:-1] = channel[:, 2:] - channel[:, :-2]

    return g_row, g_col

def gradient(image):
    g_row_by_ch = np.empty_like(image, dtype=np.float32)
    g_col_by_ch = np.empty_like(image, dtype=np.float32)
    g_mag = np.empty_like(image, dtype=np.float32)

    for idx_ch in range(image.shape[2]):
        g_row_by_ch[:, :, idx_ch], g_col_by_ch[:, :, idx_ch] = grad(image[:, :, idx_ch])
        g_mag[:, :, idx_ch] = np.hypot(g_row_by_ch[:, :, idx_ch],
                                        g_col_by_ch[:, :, idx_ch])

    # Selecting channel with highest gradient
    idcs_max = g_mag.argmax(axis=2)

    rr, cc = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij', sparse=True)
    g_row = g_row_by_ch[rr, cc, idcs_max].astype(np.float32, copy=False)
    g_col = g_col_by_ch[rr, cc, idcs_max].astype(np.float32, copy=False)

    return g_row, g_col


def compute_hog_cell(n_orientations: int, magnitudes: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    bin_width = int(180 / n_orientations)
    hog = np.zeros(n_orientations)
    # orientations = orientations%180
    for i in range(orientations.shape[0]):
        for j in range(orientations.shape[1]):
            
            orientation = orientations[i, j]
            lower_bin_index = int(orientation / bin_width)
            if lower_bin_index < 9:
                hog[lower_bin_index] += magnitudes[i, j]
            else:
                print(lower_bin_index)

    return hog / (magnitudes.shape[0] * magnitudes.shape[1])

 # Contrast Normalizing the vector
def normalize_vector(v, eps=1e-5):
    out = v / np.sqrt(np.sum(v ** 2) + eps ** 2)
    out = np.minimum(out, 0.2)
    out = out / np.sqrt(np.sum(out ** 2) + eps ** 2)
    return out


def extract_features(img, pixel_per_cell, cells_per_block, visualize=False):
    c_row, c_col = pixel_per_cell
    b_row, b_col = cells_per_block
    s_row, s_col = img.shape[:2]
    # print("S, C, B", s_row, s_col, c_row, c_col, b_row, b_col)
    
    img = img.astype(np.float64,copy=False)
    # resizedImg = cv2.resize(img, (s_row, s_col))
    resizedImg = img

    n_cells_row = int(s_row // c_row)
    n_cells_col = int(s_col // c_col) 

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1

    # Creating gradients for img ----------------------------

    g_row, g_col = gradient(resizedImg)
    grad_mag = np.hypot(g_row, g_col)
    grad_dir = np.rad2deg(np.arctan2(g_row, g_col)) % 180

    n_orientations = 9

    hog_cells = np.zeros((n_cells_row, n_cells_col, n_orientations))
    
    # print(hog_cells.shape)
    prev_x , prev_y= 0,0

    # print(grad_dir.shape, grad_mag.shape)
    
    count = 0
    tc = 0

    # print(n_cells_row, n_cells_col, n_blocks_row, n_blocks_col)

    # Compute HOG of each cell-------------------------------
    for i in range(0, s_row, c_row):
        for j in range(0, s_col, c_col):
            tc +=1
            mag_patch = grad_mag[i: i+c_row, j: j+c_col]
            ori_patch = grad_dir[i: i+c_row, j: j+c_col]
            if (mag_patch.shape != (c_row, c_col) or ori_patch.shape != (c_row, c_col)):
                count+=1
                continue
            hog_cells[i//c_row, j//c_col] = compute_hog_cell(n_orientations=n_orientations, magnitudes=mag_patch, orientations=ori_patch)
            prev_x += c_col
        prev_y += c_row
    # print(tc, count)
    hog_blocks_normalized = np.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, n_orientations))

    # Normalize HOG by block ---------------------------------
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = hog_cells[r:r + b_row, c:c + b_col, :]
            hog_blocks_normalized[r, c, :] = normalize_vector(block)


    # DRAW ---------------------------------------------------

    if visualize:
        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(n_orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / n_orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=np.float32)
        
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = draw.line(int(centre[0] - dc),
                                        int(centre[1] + dr),
                                        int(centre[0] + dc),
                                        int(centre[1] - dr))
                    hog_image[rr, cc] += hog_cells[r, c, o]
        return hog_blocks_normalized.ravel(), hog_image

    return hog_blocks_normalized.ravel()



"""
# OLD VERSION ----- for backup
def extract_features(img, pixel_per_cell, cells_per_block, visualize=False):
    c_row, c_col = pixel_per_cell
    b_row, b_col = cells_per_block
    s_row, s_col = img.shape[:2]
    # print("S, C, B", s_row, s_col, c_row, c_col, b_row, b_col)
    
    img = img.astype(np.float64,copy=False)
    # resizedImg = cv2.resize(img, (s_row, s_col))
    resizedImg = img

    n_cells_row = int(s_row // c_row)
    n_cells_col = int(s_col // c_col) 

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1

    # Creating gradients for img ----------------------------

    g_row, g_col = gradient(resizedImg)
    grad_mag = np.hypot(g_row, g_col)
    grad_dir = np.rad2deg(np.arctan2(g_row, g_col)) % 180

    n_orientations = 9

    hog_cells = np.zeros((n_cells_row, n_cells_col, n_orientations))
    
    # print(hog_cells.shape)
    prev_x = 0
    # print(n_cells_row, n_cells_col, n_blocks_row, n_blocks_col)

    # Compute HOG of each cell-------------------------------
    count = 0
    for it_x in range(n_cells_row):
        prev_y = 0
        for it_y in range(n_cells_col):
            magnitudes_patch = grad_mag[prev_y:prev_y + c_col, prev_x:prev_x + c_row]
            orientations_patch = grad_dir[prev_y:prev_y + c_col, prev_x:prev_x + c_row]
            # print(it_x, it_y)
            if (magnitudes_patch.shape != (c_row, c_col) or orientations_patch.shape != (c_row, c_col)):
                count += 1
                continue
            hog_cells[it_y, it_x] = compute_hog_cell(n_orientations, magnitudes_patch, orientations_patch)
            prev_y += c_col
        prev_x += c_row

    print(count)
    hog_blocks_normalized = np.zeros((n_blocks_row, n_blocks_col,
                                  b_row, b_col, n_orientations))

    # Normalize HOG by block ---------------------------------
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = hog_cells[r:r + b_row, c:c + b_col, :]
            hog_blocks_normalized[r, c, :] = normalize_vector(block)


    # DRAW ---------------------------------------------------

    if visualize:
        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(n_orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = (
            np.pi * (orientations_arr + .5) / n_orientations)
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=np.float32)
        
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2,
                                    c * c_col + c_col // 2])
                    rr, cc = draw.line(int(centre[0] - dc),
                                        int(centre[1] + dr),
                                        int(centre[0] + dc),
                                        int(centre[1] - dr))
                    hog_image[rr, cc] += hog_cells[r, c, o]
        return hog_blocks_normalized.ravel(), hog_image

    return hog_blocks_normalized.ravel()
"""

# DISPLAY GRADIENT -------------------------------
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 4))
# ax1.axis('off'); ax2.axis('off'); ax3.axis('off'); ax4.axis('off'); ax5.axis('off')

# ax1.imshow(g_row, cmap=plt.get_cmap('gray'))
# ax1.set_title('Gradient gx')

# ax2.imshow(g_col, cmap=plt.get_cmap('gray'))
# ax2.set_title('Gradient gy')

# ax3.imshow(grad_dir, cmap=plt.get_cmap('gray'))
# ax3.set_title('Gradient dir')

# ax4.imshow(grad_mag, cmap=plt.get_cmap('gray')) 
# ax4.set_title('Gradient mag')

# ax5.imshow(resizedImg)
# ax5.set_title('resized image')

# plt.show()