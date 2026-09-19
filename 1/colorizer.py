"""
# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
imname = 'cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(np.int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

### ag = align(g, b)
### ar = align(r, b)
# create a color image
im_out = np.dstack([ar, ag, b])

# save the image
fname = '/out_path/out_fname.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()
"""

import numpy as np
import cv2
from pathlib import Path


class Colorizer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.image.dtype == np.uint8:
            self.image = self.image.astype(np.float32) / 255.0
        elif self.image.dtype == np.uint16:
            self.image = self.image.astype(np.float32) / 65535.0
        height = self.image.shape[0] // 3
        self.blue = self.image[:height, :]
        self.green = self.image[height:2 * height, :]
        self.red = self.image[2 * height:3 * height, :]

    def shift(self, image, y_shift, x_shift):
        y_len = image.shape[0]
        x_len = image.shape[1]
        if y_shift != 0:
            if y_shift > 0:
                image_residual = image[:(y_len - y_shift), :]
                padding = np.zeros((y_shift, x_len), dtype=image.dtype)
                image = np.vstack((padding, image_residual))
            else:
                image_residual = image[-y_shift:, :]
                padding = np.zeros((-y_shift, x_len), dtype=image.dtype)
                image = np.vstack((image_residual, padding))
        
        if x_shift != 0:
            if x_shift > 0:
                image_residual = image[:, :(x_len - x_shift)]
                padding = np.zeros((y_len, x_shift), dtype=image.dtype)
                image = np.hstack((padding, image_residual))
            else:
                image_residual = image[:, -x_shift:]
                padding = np.zeros((y_len, -x_shift), dtype=image.dtype)
                image = np.hstack((image_residual, padding))
        
        return image

    def edges(self, image):
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    def crop_overlap(self, image, reference, y_shift, x_shift):
        if y_shift > 0:
            y0, y1 = y_shift, image.shape[0]
        elif y_shift < 0:
            y0, y1 = 0, image.shape[0] + y_shift
        else:
            y0, y1 = 0, image.shape[0]

        if x_shift > 0:
            x0, x1 = x_shift, image.shape[1]
        elif x_shift < 0:
            x0, x1 = 0, image.shape[1] + x_shift
        else:
            x0, x1 = 0, image.shape[1]

        return image[y0:y1, x0:x1], reference[y0:y1, x0:x1]
    
    def score_ncc(self, image_a, image_b, border_percent=0.25):
        h, w = image_a.shape
        y_crop = int(h * border_percent)
        x_crop = int(w * border_percent)

        if y_crop > 0:
            image_a = image_a[y_crop:-y_crop, :]
            image_b = image_b[y_crop:-y_crop, :]

        if x_crop > 0:
            image_a = image_a[:, x_crop:-x_crop]
            image_b = image_b[:, x_crop:-x_crop]

        if image_a.size == 0 or image_b.size == 0:
            return float('-inf')

        a = image_a.ravel()
        b = image_b.ravel()
        a = a - np.mean(a)
        b = b - np.mean(b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return float('-inf')

        return np.dot(a, b) / denominator

    def score(self, image_a, image_b, border_percent=0.25):
        h, w = image_a.shape
        y_crop = int(h * border_percent)
        x_crop = int(w * border_percent)

        if y_crop > 0:
            image_a = image_a[y_crop:-y_crop, :]
            image_b = image_b[y_crop:-y_crop, :]

        if x_crop > 0:
            image_a = image_a[:, x_crop:-x_crop]
            image_b = image_b[:, x_crop:-x_crop]

        if image_a.size == 0 or image_b.size == 0:
            return float('-inf')

        diff = image_a.astype(np.float64) - image_b.astype(np.float64)
        return -np.sum(diff * diff)
    
    def align_single(self, image, reference, search=15, use_ncc=True):
        original = image
        image = self.edges(image)
        reference = self.edges(reference)
        score_fn = self.score_ncc if use_ncc else self.score

        maxScore = float('-inf')
        max_shift = [0, 0]

        for y_shift in range(-search, search + 1):
            for x_shift in range(-search, search + 1):
                shifted_image = self.shift(image, y_shift, x_shift)
                trimmed_image, trimmed_reference = self.crop_overlap(
                    shifted_image, reference, y_shift, x_shift
                )
                currScore = score_fn(trimmed_image, trimmed_reference)
                if currScore > maxScore:
                    maxScore = currScore
                    max_shift[0] = y_shift
                    max_shift[1] = x_shift

        return self.shift(original, max_shift[0], max_shift[1]), tuple(max_shift)

    def align(self, image, reference, use_ncc=False):
        original = image
        image = self.edges(image)
        reference = self.edges(reference)
        score_fn = self.score_ncc if use_ncc else self.score

        n = 0
        max_dim = max(image.shape)
        while max_dim > 300:
            max_dim //= 2
            n += 1

        def helper(image, reference, resizes):
            if resizes == 0:
                maxScore = float('-inf')
                max_shift = [0, 0]

                for y_shift in range(-30, 31):
                    for x_shift in range(-30, 31):
                        shifted_image = self.shift(image, y_shift, x_shift)
                        trimmed_image, trimmed_reference = self.crop_overlap(
                            shifted_image, reference, y_shift, x_shift
                        )
                        currScore = score_fn(trimmed_image, trimmed_reference)
                        if currScore > maxScore:
                            maxScore = currScore
                            max_shift[0] = y_shift
                            max_shift[1] = x_shift

                return max_shift
            
            y_coarse, x_coarse = helper(
                cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                cv2.resize(reference, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                resizes - 1
            )
            y_coarse *= 2
            x_coarse *= 2
            maxScore = float('-inf')
            best_shift = [y_coarse, x_coarse]

            for y_shift in range(-5, 6):
                for x_shift in range(-5, 6):
                    candidate_y = y_coarse + y_shift
                    candidate_x = x_coarse + x_shift
                    shifted_image = self.shift(image, candidate_y, candidate_x)
                    trimmed_image, trimmed_reference = self.crop_overlap(
                        shifted_image, reference, candidate_y, candidate_x
                    )
                    currScore = score_fn(trimmed_image, trimmed_reference)
                    if currScore > maxScore:
                        maxScore = currScore
                        best_shift[0] = candidate_y
                        best_shift[1] = candidate_x

            return best_shift
        
        final_y_shift, final_x_shift = helper(image, reference, n)
        return self.shift(original, final_y_shift, final_x_shift), (final_y_shift, final_x_shift)

    def shift_is_huge_vs_g(self, red_shift, green_shift):
        ry, rx = red_shift
        gy, gx = green_shift
        r_mag = np.hypot(ry, rx)
        g_mag = max(np.hypot(gy, gx), 1.0)
        return r_mag > 4.0 * g_mag or abs(ry) > 250 or abs(rx) > 250
    
def main():
    folder = Path("CS180_fa2026_proj1_data")
    single_dir = Path("image_outputs_single")
    pyramid_dir = Path("image_outputs_pyramid")
    single_dir.mkdir(exist_ok=True)
    pyramid_dir.mkdir(exist_ok=True)

    image_paths = [
        (str(path.stem), path)
        for path in list(folder.glob("*"))
        if path.suffix.lower() in {'.jpg', '.jpeg', '.tif', '.tiff', '.png'}
    ]

    for image, image_path in image_paths:
        colorizer = Colorizer(str(image_path))
        suffix = image_path.suffix.lower()

        if suffix in {'.jpg', '.jpeg'}:
            green_output, green_shift = colorizer.align_single(colorizer.green, colorizer.blue)
            red_output, red_shift = colorizer.align_single(colorizer.red, colorizer.blue)
            print(f"[single] {image}: R->B {red_shift}, G->B {green_shift}")
            image_output = np.dstack([colorizer.blue, green_output, red_output])
            image_output = (image_output * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(single_dir / f"{image}_colorized.jpg"), image_output)

        green_output, green_shift = colorizer.align(colorizer.green, colorizer.blue)
        red_output, red_shift = colorizer.align(colorizer.red, colorizer.blue)
        if colorizer.shift_is_huge_vs_g(red_shift, green_shift):
            red_output, red_shift = colorizer.align(colorizer.red, colorizer.blue, use_ncc=True)
            print(f"[pyramid] {image}: R->B {red_shift} (NCC fallback), G->B {green_shift}")
        else:
            print(f"[pyramid] {image}: R->B {red_shift}, G->B {green_shift}")
        image_output = np.dstack([colorizer.blue, green_output, red_output])
        image_output = (image_output * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(pyramid_dir / f"{image}_colorized.jpg"), image_output)


if __name__ == "__main__":
    main()

    





    



                
                




                            







