import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.dpi'] = 70


# ## EXERCISES
#
# You are given 9 images of 400x600px, saved in the ```"images/car_{idx}.npy"``` files.

# ### 1.  Read all the images from ```./images``` and save them in a ```np.array``` (its dimension will be 9x400x600).

images = np.zeros((9, 400, 600))
for i in range(len(os.listdir('images'))):
    image = np.load(f'images/car_{i}.npy')
    images[i] = image


# ### 2. Calculate the sum of all pixels of the images.

print(np.sum(images))


# ### 3. Calculate the sum of the pixels for each image.

sums = np.array(np.sum(images, axis=(1, 2)))
print(sums)

# ### 4. Print the index of the photo having the maximum sum.

index = np.argmax(sums)
print(index)

# ### 5. Compute the mean image and show it.

mean_image = np.mean(images, axis=0)
plt.imshow(mean_image, cmap='gray')


# ### 6. Using the ```np.std(images_array)``` function, calculate the standard deviation of the images.

std_images = np.std(images)
print(std_images)


# ### 7. Normalize the images (you must subtract the mean image and divide by the standard deviation).
normalized_images = (images - mean_image) / std_images
print(normalized_images)


# ### 8. Crop each image, showing only the rows between 200 and 300 and the columns between 280 and 400.

for image in images:
    cropped_image = image[200:300, 280:400]
    plt.imshow(cropped_image, cmap='gray')
    plt.show()


# supported values for imshow cmap are:
#
# ```'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
# 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
# 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
# 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2',
# 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr',
# 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
# 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds',
# 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
# 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
# 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn',
# 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
# 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r',
# 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
# 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar',
# 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
# 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r',
# 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r',
# 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r',
# 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism',
# 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
# 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b',
# 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r',
# 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',
# 'viridis_r', 'winter', 'winter_r'```
