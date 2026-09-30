import numpy as np
import scipy.signal as signal
import cv2
from pathlib import Path
from align_image_code import align_images

class Convolutions:
    def __init__(self, image_path):
        self.image_path = image_path
        self.gray_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.color_image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        self.blue_channel = self.color_image[:, :, 0]
        self.green_channel = self.color_image[:, :, 1]
        self.red_channel = self.color_image[:, :, 2]
    
    def four_convolve(self, image, kernel):
        kernel = kernel[::-1, ::-1]
        y_pad = kernel.shape[0] // 2
        x_pad = kernel.shape[1] // 2
        image = np.hstack((np.zeros((image.shape[0], x_pad)), image, np.zeros((image.shape[0], x_pad))))
        image = np.vstack((np.zeros((y_pad, image.shape[1])), image, np.zeros((y_pad, image.shape[1]))))
        convolved_image = np.zeros(image.shape)
        for y in range(y_pad, image.shape[0] - y_pad):
            for x in range(x_pad, image.shape[1] - x_pad):
                new_value = 0
                for i in range(-y_pad, y_pad + 1):
                    for j in range(-x_pad, x_pad + 1):
                        new_value += image[y + i, x + j] * kernel[i + y_pad, j + x_pad]
                convolved_image[y, x] = new_value
        
        return convolved_image[y_pad:image.shape[0] - y_pad, x_pad:image.shape[1] - x_pad]
    
    def two_convolve(self, image, kernel):
        kernel = kernel[::-1, ::-1]
        flattened_kernel = kernel.reshape((kernel.shape[0] * kernel.shape[1]))
        y_pad = kernel.shape[0] // 2
        x_pad = kernel.shape[1] // 2
        image = np.hstack((np.zeros((image.shape[0], x_pad)), image, np.zeros((image.shape[0], x_pad))))
        image = np.vstack((np.zeros((y_pad, image.shape[1])), image, np.zeros((y_pad, image.shape[1]))))
        convolved_image = np.zeros(image.shape)
        for y in range(y_pad, image.shape[0] - y_pad):
            for x in range(x_pad, image.shape[1] - x_pad):
                new_value = 0
                extracted_image = image[y - y_pad : y + y_pad + 1, x - x_pad: x + x_pad + 1]
                extracted_image = extracted_image.reshape((extracted_image.shape[0] * extracted_image.shape[1]))
                new_value = np.dot(extracted_image, flattened_kernel)
                convolved_image[y, x] = new_value
        
        return convolved_image[y_pad:image.shape[0] - y_pad, x_pad:image.shape[1] - x_pad]
    
    def unsharp_mask_filter(self, image, kernel, alpha):
        image = image.astype(np.float64)
        blurred_image = self.four_convolve(image, kernel)

        high_frequency_image = image - blurred_image

        sharpened_image = image + alpha * high_frequency_image
        sharpened_image_rescaled = np.clip(np.round(sharpened_image), 0, 255).astype(np.uint8)
        return sharpened_image_rescaled

    def unsharp_mask_single_filter(self, image, kernel, alpha):
        image = image.astype(np.float64)
        impulse = np.zeros(kernel.shape)
        impulse[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1
        unsharp_kernel = (1 + alpha) * impulse - alpha * kernel

        sharpened_image = self.four_convolve(image, unsharp_kernel)
        sharpened_image_rescaled = np.clip(np.round(sharpened_image), 0, 255).astype(np.uint8)
        return sharpened_image_rescaled
        

    

def part_1_1():
    convolutions = Convolutions("selfie_1.jpg")
    d_x = np.array([[1, 0, -1]])
    d_y = np.array([[1], [0], [-1]])
    box_filter = np.ones((9, 9)) / 81

    finite_four_convolved_image_custom_x = convolutions.four_convolve(convolutions.gray_image, d_x)
    max_x = np.max(np.abs(finite_four_convolved_image_custom_x))
    finite_four_convolved_image_custom_x_rescaled = ((finite_four_convolved_image_custom_x + max_x) * (255 / (2 * max_x))).astype(np.uint8)
    finite_four_convolved_image_custom_y = convolutions.four_convolve(convolutions.gray_image, d_y)
    max_y = np.max(np.abs(finite_four_convolved_image_custom_y))
    finite_four_convolved_image_custom_y_rescaled = ((finite_four_convolved_image_custom_y + max_y) * (255 / (2 * max_y))).astype(np.uint8)
    finite_four_convolved_image_custom = np.sqrt(np.square(finite_four_convolved_image_custom_x) + np.square(finite_four_convolved_image_custom_y))
    max_overall = np.max(finite_four_convolved_image_custom)
    finite_four_convolved_image_custom_rescaled = (finite_four_convolved_image_custom * (255 / max_overall)).astype(np.uint8)

    box_filter_custom = convolutions.four_convolve(convolutions.gray_image, box_filter)
    box_filter_custom_rescaled = np.clip(np.round(box_filter_custom), 0, 255).astype(np.uint8)

    finite_scipy = signal.convolve2d(convolutions.gray_image, d_x, mode='same')
    max_scipy_x = np.max(np.abs(finite_scipy))
    finite_scipy_x_rescaled = ((finite_scipy + max_scipy_x) * (255 / (2 * max_scipy_x))).astype(np.uint8)
    finite_scipy_y = signal.convolve2d(convolutions.gray_image, d_y, mode='same')
    max_scipy_y = np.max(np.abs(finite_scipy_y))
    finite_scipy_y_rescaled = ((finite_scipy_y + max_scipy_y) * (255 / (2 * max_scipy_y))).astype(np.uint8)
    finite_scipy_magnitude = np.sqrt(np.square(finite_scipy) + np.square(finite_scipy_y))
    max_scipy_overall = np.max(finite_scipy_magnitude)
    finite_scipy_magnitude_rescaled = (finite_scipy_magnitude * (255 / max_scipy_overall)).astype(np.uint8)

    box_filter_scipy = signal.convolve2d(convolutions.gray_image, box_filter, mode='same')
    box_filter_scipy_rescaled = np.clip(np.round(box_filter_scipy), 0, 255).astype(np.uint8)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "dx_custom_vs_scipy.jpg"), np.hstack((finite_four_convolved_image_custom_x_rescaled, finite_scipy_x_rescaled)))
    cv2.imwrite(str(output_dir / "dy_custom_vs_scipy.jpg"), np.hstack((finite_four_convolved_image_custom_y_rescaled, finite_scipy_y_rescaled)))
    cv2.imwrite(str(output_dir / "magnitude_custom_vs_scipy.jpg"), np.hstack((finite_four_convolved_image_custom_rescaled, finite_scipy_magnitude_rescaled)))
    cv2.imwrite(str(output_dir / "box_custom_vs_scipy.jpg"), np.hstack((box_filter_custom_rescaled, box_filter_scipy_rescaled)))

def part_1_2():
    cameraman = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
    d_x = np.array([[1, 0, -1]])
    d_y = np.array([[1], [0], [-1]])
    finite_scipy = signal.convolve2d(cameraman, d_x, mode='same')
    max_scipy_x = np.max(np.abs(finite_scipy))
    finite_scipy_x_rescaled = ((finite_scipy + max_scipy_x) * (255 / (2 * max_scipy_x))).astype(np.uint8)
    finite_scipy_y = signal.convolve2d(cameraman, d_y, mode='same')
    max_scipy_y = np.max(np.abs(finite_scipy_y))
    finite_scipy_y_rescaled = ((finite_scipy_y + max_scipy_y) * (255 / (2 * max_scipy_y))).astype(np.uint8)
    finite_scipy_magnitude = np.sqrt(np.square(finite_scipy) + np.square(finite_scipy_y))
    max_scipy_overall = np.max(finite_scipy_magnitude)
    finite_scipy_magnitude_rescaled = (finite_scipy_magnitude * (255 / max_scipy_overall)).astype(np.uint8)

    threshold = 25
    finite_scipy_magnitude_edged = np.where(finite_scipy_magnitude_rescaled > threshold, 255, 0).astype(np.uint8)
    
    output_dir = Path("1.2")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "dx.jpg"), finite_scipy_x_rescaled)
    cv2.imwrite(str(output_dir / "dy.jpg"), finite_scipy_y_rescaled)
    cv2.imwrite(str(output_dir / "magnitude.jpg"), finite_scipy_magnitude_rescaled)
    cv2.imwrite(str(output_dir / "original_vs_edged.jpg"), np.hstack((cameraman, finite_scipy_magnitude_edged)))

    for trial_threshold in [30, 35, 40, 45, 50]:
        trial_edged = np.where(finite_scipy_magnitude_rescaled > trial_threshold, 255, 0).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"original_vs_edged_threshold_{trial_threshold}.jpg"), np.hstack((cameraman, trial_edged)))


def part_1_3():
    cameraman = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
    oned_gaussian_kernel = cv2.getGaussianKernel(13, 2)
    gaussian_kernel = oned_gaussian_kernel @ oned_gaussian_kernel.T
    gaussian_cameraman = signal.convolve2d(cameraman, gaussian_kernel, mode='same')
    max_gaussian_cameraman = np.max(gaussian_cameraman)
    gaussian_cameraman_rescaled = (gaussian_cameraman * (255 / max_gaussian_cameraman)).astype(np.uint8)

    d_x = np.array([[1, 0, -1]])
    d_y = np.array([[1], [0], [-1]])
    gaussian_x = signal.convolve2d(gaussian_cameraman, d_x, mode='same')
    gaussian_y = signal.convolve2d(gaussian_cameraman, d_y, mode='same')
    gaussian_magnitude = np.sqrt(np.square(gaussian_x) + np.square(gaussian_y))
    max_gaussian_magnitude = np.max(gaussian_magnitude)
    gaussian_magnitude_rescaled = (gaussian_magnitude * (255 / max_gaussian_magnitude)).astype(np.uint8)

    threshold = 40
    gaussian_magnitude_edged = np.where(gaussian_magnitude_rescaled > threshold, 255, 0).astype(np.uint8)

    output_dir = Path("1.3")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "blurred.jpg"), gaussian_cameraman_rescaled)
    cv2.imwrite(str(output_dir / "blurred_magnitude_vs_edged.jpg"), np.hstack((gaussian_magnitude_rescaled, gaussian_magnitude_edged)))

    convolved_gaussian_filter_x = signal.convolve2d(gaussian_kernel, d_x, mode='full')
    convolved_gaussian_filter_y = signal.convolve2d(gaussian_kernel, d_y, mode='full')
    for filter_name, dog_filter in [("gaussian_filter", gaussian_kernel), ("dog_filter_x", convolved_gaussian_filter_x), ("dog_filter_y", convolved_gaussian_filter_y)]:
        max_filter = np.max(np.abs(dog_filter))
        dog_filter_rescaled = ((dog_filter + max_filter) * (255 / (2 * max_filter))).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{filter_name}.png"), cv2.resize(dog_filter_rescaled, None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST))
    cameraman_gaussian_x = signal.convolve2d(cameraman, convolved_gaussian_filter_x, mode='same')
    cameraman_gaussian_y = signal.convolve2d(cameraman, convolved_gaussian_filter_y, mode='same')
    cameraman_gaussian_magnitude = np.sqrt(np.square(cameraman_gaussian_x) + np.square(cameraman_gaussian_y))
    cameraman_gaussian_magnitude_rescaled = (cameraman_gaussian_magnitude * (255 / max_gaussian_magnitude)).astype(np.uint8)
    cameraman_gaussian_magnitude_edged = np.where(cameraman_gaussian_magnitude_rescaled > threshold, 255, 0).astype(np.uint8)

    output_dir = Path("1.3")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "cameraman_gaussian_magnitude_vs_edged.jpg"), np.hstack((cameraman_gaussian_magnitude_rescaled, cameraman_gaussian_magnitude_edged)))

def part_2_1():
    convolutions = Convolutions("taj.jpg")
    oned_gaussian_kernel = cv2.getGaussianKernel(9, 1.5)
    gaussian_kernel = oned_gaussian_kernel @ oned_gaussian_kernel.T

    def color_unsharp_mask_filter(image, alpha):
        channels = [convolutions.unsharp_mask_single_filter(image[:, :, c], gaussian_kernel, alpha) for c in range(3)]
        return np.dstack(channels)

    output_dir = Path("2.1")
    output_dir.mkdir(exist_ok=True)

    taj = convolutions.color_image
    taj_two_step = np.dstack([convolutions.unsharp_mask_filter(taj[:, :, c], gaussian_kernel, 1) for c in range(3)])
    taj_single_filter = color_unsharp_mask_filter(taj, 1)
    cv2.imwrite(str(output_dir / "taj_two_step_vs_single_filter.jpg"), np.hstack((taj_two_step, taj_single_filter)))

    taj_sharpened = [color_unsharp_mask_filter(taj, alpha) for alpha in [0.5, 1, 2, 4]]
    cv2.imwrite(str(output_dir / "taj_original_vs_sharpened_alphas.jpg"), np.hstack([taj] + taj_sharpened))

    selfie = cv2.imread("selfie_1.jpg", cv2.IMREAD_COLOR)
    selfie = cv2.resize(selfie, (450, 600), interpolation=cv2.INTER_AREA)
    selfie_sharpened = color_unsharp_mask_filter(selfie, 2)
    cv2.imwrite(str(output_dir / "selfie_original_vs_sharpened.jpg"), np.hstack((selfie, selfie_sharpened)))

    for name, image, alpha in [("taj", taj, 1), ("selfie", selfie, 2)]:
        blurred = np.dstack([convolutions.four_convolve(image[:, :, c].astype(np.float64), gaussian_kernel) for c in range(3)])
        high_frequency = image - blurred
        cv2.imwrite(str(output_dir / f"{name}_blurred.jpg"), np.clip(np.round(blurred), 0, 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{name}_high_frequency.jpg"), np.clip(np.round(high_frequency * 2 + 128), 0, 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{name}_sharpened.jpg"), color_unsharp_mask_filter(image, alpha))

    cameraman = cv2.imread("cameraman.jpg", cv2.IMREAD_GRAYSCALE)
    cameraman = cv2.resize(cameraman, (543, 540), interpolation=cv2.INTER_AREA)
    cameraman_blurred = np.clip(np.round(convolutions.four_convolve(cameraman, gaussian_kernel)), 0, 255).astype(np.uint8)
    cameraman_resharpened = convolutions.unsharp_mask_single_filter(cameraman_blurred, gaussian_kernel, 2)
    cv2.imwrite(str(output_dir / "cameraman_original_vs_blurred_vs_resharpened.jpg"), np.hstack((cameraman, cameraman_blurred, cameraman_resharpened)))

def gaussian_convolve(image, sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    oned_gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
    channels = []
    for c in range(image.shape[2]):
        blurred_channel = signal.convolve2d(image[:, :, c], oned_gaussian_kernel, mode='same', boundary='symm')
        blurred_channel = signal.convolve2d(blurred_channel, oned_gaussian_kernel.T, mode='same', boundary='symm')
        channels.append(blurred_channel)
    return np.dstack(channels)

def float_rgb_imread(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1] / 255

def uint8_bgr_rescale(image, height=625):
    image = np.ascontiguousarray(np.clip(np.round(image[:, :, ::-1] * 255), 0, 255).astype(np.uint8))
    width = int(round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def hybrid_filter(low_image, high_image, sigma_low, sigma_high):
    low_image = low_image.astype(np.float64)
    high_image = high_image.astype(np.float64)
    low_frequencies = gaussian_convolve(low_image, sigma_low)
    high_frequencies = high_image - gaussian_convolve(high_image, sigma_high)
    hybrid = np.clip(np.round(low_frequencies + high_frequencies), 0, 255).astype(np.uint8)
    return low_frequencies, high_frequencies, hybrid

def log_fft_magnitude(image):
    image = image.astype(np.float64)
    gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
    log_magnitude = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    log_magnitude_rescaled = ((log_magnitude - np.min(log_magnitude)) * (255 / (np.max(log_magnitude) - np.min(log_magnitude)))).astype(np.uint8)
    return cv2.applyColorMap(log_magnitude_rescaled, cv2.COLORMAP_VIRIDIS)

def label_image(image, label):
    banner = np.full((40, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(banner, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((banner, image))

def multiscale_hstack(image, levels=4):
    tiles = [image]
    for level in range(1, levels):
        shrunk = cv2.resize(image, None, fx=0.5 ** level, fy=0.5 ** level, interpolation=cv2.INTER_AREA)
        padding = np.full((image.shape[0] - shrunk.shape[0], shrunk.shape[1], 3), 255, dtype=np.uint8)
        tiles.append(np.vstack((padding, shrunk)))
    return np.hstack(tiles)

def part_2_2():
    output_dir = Path("2.2")
    output_dir.mkdir(exist_ok=True)
    examples = [
        ("derek_nutmeg", "derek.jpg", "nutmeg.jpg", 6, 3),
        ("lion_tiger", "lion.jpg", "tiger.jpg", 8, 2),
        ("selfie_lion", "selfie_1.jpg", "lion.jpg", 12, 4),
    ]
    favorite = "selfie_lion"

    for name, low_path, high_path, sigma_low, sigma_high in examples:
        high_image, low_image = align_images(float_rgb_imread(high_path), float_rgb_imread(low_path))
        low_image = uint8_bgr_rescale(low_image)
        high_image = uint8_bgr_rescale(high_image)
        low_frequencies, high_frequencies, hybrid = hybrid_filter(low_image, high_image, sigma_low, sigma_high)
        cv2.imwrite(str(output_dir / f"{name}_inputs_vs_hybrid.jpg"), np.hstack((label_image(low_image, "Low-pass input"), label_image(high_image, "High-pass input"), label_image(hybrid, "Hybrid"))))
        cv2.imwrite(str(output_dir / f"{name}_hybrid_multiscale.jpg"), multiscale_hstack(hybrid))

        if name == favorite:
            low_frequencies_rescaled = np.clip(np.round(low_frequencies), 0, 255).astype(np.uint8)
            high_frequencies_rescaled = np.clip(np.round(high_frequencies + 128), 0, 255).astype(np.uint8)
            images_row = np.hstack((label_image(low_image, "Low input"), label_image(high_image, "High input"), label_image(low_frequencies_rescaled, "Low-passed"), label_image(high_frequencies_rescaled, "High-passed"), label_image(hybrid, "Hybrid")))
            fft_row = np.hstack((label_image(log_fft_magnitude(low_image), "FFT low input"), label_image(log_fft_magnitude(high_image), "FFT high input"), label_image(log_fft_magnitude(low_frequencies), "FFT low-passed"), label_image(log_fft_magnitude(high_frequencies), "FFT high-passed"), label_image(log_fft_magnitude(hybrid), "FFT hybrid")))
            frequency_analysis = cv2.resize(np.vstack((images_row, fft_row)), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(output_dir / f"{name}_frequency_analysis.jpg"), frequency_analysis)

def gaussian_stack(image, levels, sigma):
    stack = [image.astype(np.float64)]
    for level in range(1, levels):
        stack.append(gaussian_convolve(stack[-1], sigma * 2 ** (level - 1)))
    return stack

def laplacian_stack(image, levels, sigma):
    gaussian = gaussian_stack(image, levels, sigma)
    stack = []
    for level in range(levels - 1):
        stack.append(gaussian[level] - gaussian[level + 1])
    stack.append(gaussian[-1])
    return stack

def stack_blend(image_a, image_b, mask, levels, sigma):
    laplacian_a = laplacian_stack(image_a, levels, sigma)
    laplacian_b = laplacian_stack(image_b, levels, sigma)
    gaussian_mask = gaussian_stack(mask, levels, sigma)
    masked_a, masked_b, blended = [], [], []
    for level in range(levels):
        masked_a.append(gaussian_mask[level] * laplacian_a[level])
        masked_b.append((1 - gaussian_mask[level]) * laplacian_b[level])
        blended.append(masked_a[level] + masked_b[level])
    return masked_a, masked_b, blended

def normalize_rescale(image):
    return ((image - np.min(image)) * (255 / (np.max(image) - np.min(image)))).astype(np.uint8)

def clip_rescale(image):
    return np.clip(np.round(image), 0, 255).astype(np.uint8)

def part_2_3():
    output_dir = Path("2.3")
    output_dir.mkdir(exist_ok=True)
    levels = 5
    sigma = 2
    apple = cv2.imread("apple.jpeg", cv2.IMREAD_COLOR)
    orange = cv2.imread("orange.jpeg", cv2.IMREAD_COLOR)
    mask = np.zeros(apple.shape)
    mask[:, :apple.shape[1] // 2] = 1

    for name, image in [("apple", apple), ("orange", orange)]:
        gaussian_row = np.hstack([label_image(clip_rescale(level), f"G{i}") for i, level in enumerate(gaussian_stack(image, levels, sigma))])
        laplacian_row = np.hstack([label_image(normalize_rescale(level), f"L{i}") for i, level in enumerate(laplacian_stack(image, levels, sigma))])
        cv2.imwrite(str(output_dir / f"{name}_gaussian_and_laplacian_stacks.jpg"), np.vstack((gaussian_row, laplacian_row)))

    masked_apple, masked_orange, blended = stack_blend(apple, orange, mask, levels, sigma)
    cv2.imwrite(str(output_dir / "oraple_figure_3_42.jpg"), blend_figure(masked_apple, masked_orange, blended, "Apple", "Orange", "Oraple"))
    cv2.imwrite(str(output_dir / "oraple.jpg"), clip_rescale(sum(blended)))

def blend_figure(masked_a, masked_b, blended, name_a, name_b, name_blend):
    rows = []
    for level in [0, 2, 4]:
        rows.append(np.hstack((label_image(normalize_rescale(masked_a[level]), f"{name_a} level {level}"), label_image(normalize_rescale(masked_b[level]), f"{name_b} level {level}"), label_image(normalize_rescale(blended[level]), f"Blend level {level}"))))
    rows.append(np.hstack((label_image(clip_rescale(sum(masked_a)), f"{name_a} collapsed"), label_image(clip_rescale(sum(masked_b)), f"{name_b} collapsed"), label_image(clip_rescale(sum(blended)), name_blend))))
    return np.vstack(rows)

def aligned_pair_imread(path_1, path_2):
    image_1, image_2 = align_images(float_rgb_imread(path_1), float_rgb_imread(path_2))
    return uint8_bgr_rescale(image_1), uint8_bgr_rescale(image_2)

def elliptical_mask(shape, ellipses):
    mask = np.zeros(shape[:2])
    for (center_x, center_y), (axis_x, axis_y) in ellipses:
        cv2.ellipse(mask, (int(center_x * shape[1]), int(center_y * shape[0])), (int(axis_x * shape[1]), int(axis_y * shape[0])), 0, 0, 360, 1, -1)
    return np.dstack([mask] * 3)

def part_2_4():
    output_dir = Path("2.4")
    output_dir.mkdir(exist_ok=True)
    levels = 6
    sigma = 2

    apple = cv2.imread("apple.jpeg", cv2.IMREAD_COLOR)
    orange = cv2.imread("orange.jpeg", cv2.IMREAD_COLOR)
    oraple_mask = np.zeros(apple.shape)
    oraple_mask[:, :apple.shape[1] // 2] = 1
    blended = stack_blend(apple, orange, oraple_mask, levels, sigma)[2]
    cv2.imwrite(str(output_dir / "oraple_inputs_vs_blend.jpg"), np.hstack((apple, orange, clip_rescale(sum(blended)))))

    tiger, lion = aligned_pair_imread("tiger.jpg", "lion.jpg")
    liger_mask = np.zeros(lion.shape)
    liger_mask[:, :lion.shape[1] // 2] = 1
    blended = stack_blend(lion, tiger, liger_mask, levels, sigma)[2]
    cv2.imwrite(str(output_dir / "liger_inputs_vs_blend.jpg"), np.hstack((label_image(lion, "Lion"), label_image(tiger, "Tiger"), label_image(clip_rescale(sum(blended)), "Liger"))))

    selfie, lion = aligned_pair_imread("selfie_1.jpg", "lion.jpg")
    face_mask = elliptical_mask(selfie.shape, [((0.5, 0.57), (0.28, 0.24))])
    masked_selfie, masked_lion, blended = stack_blend(selfie, lion, face_mask, levels, sigma)
    cv2.imwrite(str(output_dir / "selfie_in_lion_inputs_vs_blend.jpg"), np.hstack((label_image(selfie, "Selfie"), label_image(lion, "Lion"), label_image(clip_rescale(face_mask * 255), "Mask"), label_image(clip_rescale(sum(blended)), "Blend"))))
    cv2.imwrite(str(output_dir / "selfie_in_lion_laplacian_figure.jpg"), blend_figure(masked_selfie, masked_lion, blended, "Selfie", "Lion", "Blend"))

    tiger, selfie = aligned_pair_imread("tiger.jpg", "selfie_1.jpg")
    eyes_mask = elliptical_mask(selfie.shape, [((0.371, 0.5), (0.085, 0.045)), ((0.629, 0.5), (0.085, 0.045))])
    blended = stack_blend(tiger, selfie, eyes_mask, levels, sigma)[2]
    cv2.imwrite(str(output_dir / "tiger_eyes_inputs_vs_blend.jpg"), np.hstack((label_image(tiger, "Tiger"), label_image(selfie, "Selfie"), label_image(clip_rescale(eyes_mask * 255), "Mask"), label_image(clip_rescale(sum(blended)), "Blend"))))

def main():
    part_2_4()


if __name__ == "__main__":
    main()
    
    


