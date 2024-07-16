# Image Compression with K-Means Clustering

This project demonstrates image compression using K-Means clustering. The code allows you to compress an image by reducing the number of colors and then reconstruct the image using the compressed color palette.

## Original Image
![Original Image](https://github.com/Mohd-Monis/K-Means-Image-Compressor/blob/main/lion3.jpeg)

## Compressed Image
![Compressed Image](https://github.com/Mohd-Monis/K-Means-Image-Compressor/blob/main/compressed_preview.PNG)

## Compressed 19KB file to 14KB
![Compressoin](https://github.com/Mohd-Monis/K-Means-Image-Compressor/blob/main/Compression.png)

## Features

- Compress an image by reducing the number of colors.
- Reconstruct and display the compressed image.
- Save the compressed image data for future use.

## Requirements

- Python 3.x
- NumPy
- Pillow (PIL)

### Compress an Image

1. Run the script:
    ```bash
    python k_nearest.py
    ```
2. Select the option to compress an image.
3. Enter the path to the image and the desired number of colors (K).

### Open a Compressed Image

1. Run the script:
    ```bash
    python k_nearest.py
    ```
2. Select the option to open a compressed image.
3. Enter the path to the original image.

## How It Works

1. **K-Means Initialization**: The initial centroids are chosen randomly from the image pixels.
2. **Assignment Step**: Each pixel is assigned to the nearest centroid.
3. **Update Step**: The centroids are updated to be the mean of the pixels assigned to them.
4. **Reconstruction**: The image is reconstructed using the compressed color palette.

## Limitations

- The compression quality is dependent on the chosen value of K (number of colors).
- Large images may take longer to process.

## Example

Here is an example of how to use the script:

```python
import numpy as np
from PIL import Image
import random
import sys

# Example usage
run_k_means()
