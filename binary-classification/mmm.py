from PIL import Image
import os
import matplotlib.pyplot as plt

def get_min_max_dimensions(folder_path):
    min_width = float('inf')
    min_height = float('inf')
    max_width = 0
    max_height = 0
    min_width_image = ''
    min_height_image = ''
    max_width_image = ''
    max_height_image = ''

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                image = Image.open(file_path)
                width, height = image.size
                if width < min_width:
                    min_width = width
                    min_width_image = file_path
                if height < min_height:
                    min_height = height
                    min_height_image = file_path
                if width > max_width:
                    max_width = width
                    max_width_image = file_path
                if height > max_height:
                    max_height = height
                    max_height_image = file_path
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return min_width_image, min_height_image, max_width_image, max_height_image

folder_path = './data/Images'
min_width_image, min_height_image, max_width_image, max_height_image = get_min_max_dimensions(folder_path)

print(f"Minimum Width Image: {min_width_image}")
print(f"Minimum Height Image: {min_height_image}")
print(f"Maximum Width Image: {max_width_image}")
print(f"Maximum Height Image: {max_height_image}")

# Plotting images
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(Image.open(min_width_image).resize((224, 224)))
axes[0, 0].set_title(f"Min Width Image\nWidth: {min_width_image}, Height: {min_height_image}")

axes[0, 1].imshow(Image.open(min_height_image).resize((224, 224)))
axes[0, 1].set_title(f"Min Height Image\nWidth: {min_width_image}, Height: {min_height_image}")

axes[1, 0].imshow(Image.open(max_width_image).resize((224, 224)))
axes[1, 0].set_title(f"Max Width Image\nWidth: {max_width_image}, Height: {max_height_image}")

axes[1, 1].imshow(Image.open(max_height_image).resize((224, 224)))
axes[1, 1].set_title(f"Max Height Image\nWidth: {max_width_image}, Height: {max_height_image}")

for ax in axes.flatten():
    ax.axis('off')

plt.show()
