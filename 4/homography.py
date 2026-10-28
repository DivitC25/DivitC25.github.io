import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from pathlib import Path

def build_system(im1_pts, im2_pts):
    im1_pts = np.asarray(im1_pts, dtype=np.float64)
    im2_pts = np.asarray(im2_pts, dtype=np.float64)
    num_points = im1_pts.shape[0]
    system = np.zeros((2 * num_points, 8))
    targets = np.zeros(2 * num_points)
    for i in range(num_points):
        x, y = im1_pts[i]
        x_prime, y_prime = im2_pts[i]
        system[2 * i] = [x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime]
        system[2 * i + 1] = [0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime]
        targets[2 * i] = x_prime
        targets[2 * i + 1] = y_prime
    return system, targets

def computeH(im1_pts, im2_pts):
    system, targets = build_system(im1_pts, im2_pts)
    solution, _, _, _ = np.linalg.lstsq(system, targets, rcond=None)
    homography = np.array([
        [solution[0], solution[1], solution[2]],
        [solution[3], solution[4], solution[5]],
        [solution[6], solution[7], 1.0],
    ])
    return homography

def apply_homography(homography, points):
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    mapped = (homography @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]

def image_corners(image):
    height, width = image.shape[:2]
    return np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float64)

def warp_bounds(image, homography):
    mapped = apply_homography(homography, image_corners(image))
    minimum = np.floor(np.min(mapped, axis=0)).astype(int)
    maximum = np.ceil(np.max(mapped, axis=0)).astype(int)
    return minimum, maximum

def inverse_coordinates(minimum, maximum, homography):
    xs = np.arange(minimum[0], maximum[0] + 1)
    ys = np.arange(minimum[1], maximum[1] + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    source = apply_homography(np.linalg.inv(homography), flat)
    return source[:, 0].reshape(grid_x.shape), source[:, 1].reshape(grid_y.shape)

def sample_nearest(image, source_x, source_y):
    height, width = image.shape[:2]
    nearest_x = np.floor(source_x + 0.5).astype(np.int32)
    nearest_y = np.floor(source_y + 0.5).astype(np.int32)
    valid = (nearest_x >= 0) & (nearest_y >= 0) & (nearest_x < width) & (nearest_y < height)
    colors = np.zeros(source_x.shape + (image.shape[2],), dtype=np.uint8)
    colors[valid] = image[nearest_y[valid], nearest_x[valid]]
    alpha = np.zeros(source_x.shape, dtype=np.uint8)
    alpha[valid] = 255
    return colors, alpha

def sample_bilinear(image, source_x, source_y):
    height, width = image.shape[:2]
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    dx = (source_x - x0).astype(np.float32)
    dy = (source_y - y0).astype(np.float32)
    weight_00 = (1 - dx) * (1 - dy)
    weight_10 = dx * (1 - dy)
    weight_01 = (1 - dx) * dy
    weight_11 = dx * dy
    image_f = image.astype(np.float32)
    colors = np.zeros(source_x.shape + (image.shape[2],), dtype=np.float32)
    colors[valid] = (
        image_f[y0[valid], x0[valid]] * weight_00[valid, None]
        + image_f[y0[valid], x1[valid]] * weight_10[valid, None]
        + image_f[y1[valid], x0[valid]] * weight_01[valid, None]
        + image_f[y1[valid], x1[valid]] * weight_11[valid, None]
    )
    alpha = np.zeros(source_x.shape, dtype=np.uint8)
    alpha[valid] = 255
    return np.clip(np.round(colors), 0, 255).astype(np.uint8), alpha

def warp_image(image, homography, sample):
    minimum, maximum = warp_bounds(image, homography)
    source_x, source_y = inverse_coordinates(minimum, maximum, homography)
    colors, alpha = sample(image, source_x, source_y)
    warped = np.dstack([colors, alpha])
    return warped, minimum

def warpImageNearestNeighbor(image, homography):
    warped, _ = warp_image(image, homography, sample_nearest)
    return warped

def warpImageBilinear(image, homography):
    warped, _ = warp_image(image, homography, sample_bilinear)
    return warped

def rectangle_from_corners(corners):
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    width = int(round((top + bottom) / 2))
    height = int(round((left + right) / 2))
    return np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float64)

def composite_warp(warped):
    background = np.full(warped.shape[:2] + (3,), 244, dtype=np.float32)
    alpha = warped[:, :, 3:4].astype(np.float32) / 255
    merged = warped[:, :, :3].astype(np.float32) * alpha + background * (1 - alpha)
    return np.clip(np.round(merged), 0, 255).astype(np.uint8)

def draw_correspondences(image_1, image_2, points_1, points_2, output_path):
    height, width = image_1.shape[:2]
    gap = 28
    canvas = np.full((height, width * 2 + gap, 3), 244, np.uint8)
    canvas[:, :width] = image_1
    canvas[:, width + gap:] = image_2

    figure, axis = plt.subplots(figsize=(14.2, 5.5))
    axis.imshow(canvas)
    colors = plt.cm.tab20(np.linspace(0, 1, len(points_1)))
    stroke = [pe.withStroke(linewidth=2.6, foreground="black")]
    for i, ((x, y), (x_prime, y_prime), color) in enumerate(zip(points_1, points_2, colors), start=1):
        axis.plot([x, x_prime + width + gap], [y, y_prime], color=color, lw=1.3, solid_capstyle="round")
        axis.scatter([x, x_prime + width + gap], [y, y_prime], s=36, color=[color], edgecolors="white", linewidths=0.7, zorder=3)
        axis.text(x + 7, y - 8, str(i), color="white", fontsize=8, fontweight="bold", path_effects=stroke, zorder=4)
        axis.text(x_prime + width + gap + 7, y_prime - 8, str(i), color="white", fontsize=8, fontweight="bold", path_effects=stroke, zorder=4)
    axis.set_axis_off()
    figure.tight_layout(pad=0.15)
    figure.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)

def draw_reprojection(image_2, points_2, projected_points, output_path):
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.imshow(image_2)
    axis.scatter(points_2[:, 0], points_2[:, 1], s=56, facecolors="none", edgecolors="#5ee0ff", linewidths=1.5, label="measured point", zorder=3)
    axis.scatter(projected_points[:, 0], projected_points[:, 1], s=32, c="#ffe14a", marker="x", linewidths=1.3, label="point mapped by H", zorder=4)
    stroke = [pe.withStroke(linewidth=2.6, foreground="black")]
    for i, (x, y) in enumerate(points_2, start=1):
        axis.text(x + 8, y - 8, str(i), color="white", fontsize=8, fontweight="bold", path_effects=stroke, zorder=5)
    axis.legend(loc="lower right", fontsize=8, framealpha=0.92)
    axis.set_axis_off()
    figure.tight_layout(pad=0.15)
    figure.savefig(output_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)

def format_system(system, targets):
    header = " ".join(f"{name:>14}" for name in ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32"]) + f"  {'b':>12}"
    lines = [header]
    for row, target in zip(system, targets):
        lines.append(" ".join(f"{value:14.4f}" for value in row) + f"  {target:12.4f}")
    return "\n".join(lines)

def format_homography(homography):
    return "\n".join("  ".join(f"{value:12.6f}" for value in row) for row in homography)

def save_pair(name, image_1_path, image_2_path, points_1, points_2, output_dir):
    image_1 = cv2.cvtColor(cv2.imread(str(image_1_path)), cv2.COLOR_BGR2RGB)
    image_2 = cv2.cvtColor(cv2.imread(str(image_2_path)), cv2.COLOR_BGR2RGB)
    homography = computeH(points_1, points_2)
    system, targets = build_system(points_1, points_2)
    projected_points = apply_homography(homography, points_1)
    error = np.linalg.norm(projected_points - points_2, axis=1)

    lines = [name, f"mean {error.mean():.4f}  max {error.max():.4f}", "per-point"]
    for i, point_error in enumerate(error, start=1):
        lines.append(f"{i:2d}  {point_error:.4f}")
    lines.append("H")
    lines.append(format_homography(homography))
    lines.append("Ab")
    lines.append(format_system(system, targets))
    (output_dir / f"{name}_system.txt").write_text("\n".join(lines) + "\n")

    draw_correspondences(image_1, image_2, points_1, points_2, output_dir / f"{name}_corr.jpg")
    draw_reprojection(image_2, points_2, projected_points, output_dir / f"{name}_reproj.jpg")
    print(name, f"mean {error.mean():.3f}px", f"max {error.max():.3f}px")

def check_synthetic():
    rng = np.random.default_rng(0)
    true_homography = np.array([
        [0.90, 0.05, 40.0],
        [-0.02, 0.95, 30.0],
        [1e-4, -8e-5, 1.0],
    ])
    points = rng.uniform([50, 50], [1500, 1100], size=(12, 2))
    destination = apply_homography(true_homography, points)
    recovered = computeH(points, destination)
    error = np.linalg.norm(apply_homography(recovered, points) - destination, axis=1)
    if error.max() > 1e-6:
        raise RuntimeError(f"synthetic homography recovery failed, max error {error.max()}")

def part_a_2():
    check_synthetic()
    root = Path(__file__).resolve().parent
    output_dir = root / "a2"
    output_dir.mkdir(exist_ok=True)

    living_room_1 = np.array([
        [530.20, 514.50],
        [1289.08, 558.40],
        [615.29, 932.67],
        [1460.07, 997.67],
        [1038.93, 891.13],
        [812.51, 621.47],
        [1241.68, 984.77],
        [526.88, 759.42],
        [1110.99, 484.10],
        [1127.25, 721.19],
        [957.88, 617.29],
        [865.63, 752.67],
    ])
    living_room_2 = np.array([
        [168.38, 513.11],
        [968.28, 585.37],
        [275.46, 1008.82],
        [1092.99, 955.53],
        [755.63, 904.02],
        [528.59, 641.39],
        [930.60, 970.82],
        [154.38, 811.80],
        [823.56, 513.43],
        [835.69, 734.55],
        [681.02, 638.30],
        [582.66, 778.34],
    ])
    posters_1 = np.array([
        [668.84, 259.57],
        [1573.50, 301.56],
        [744.58, 713.75],
        [1529.80, 1146.58],
        [1196.57, 589.57],
        [1052.36, 254.27],
        [1205.19, 1016.47],
        [868.98, 456.01],
        [1488.61, 536.23],
        [1286.74, 379.31],
        [635.42, 484.95],
        [998.26, 634.44],
    ])
    posters_2 = np.array([
        [58.81, 255.50],
        [960.21, 371.43],
        [165.09, 784.35],
        [959.65, 1110.71],
        [662.44, 624.60],
        [511.69, 291.43],
        [682.13, 1043.44],
        [316.84, 488.77],
        [904.43, 572.34],
        [737.32, 427.09],
        [11.10, 518.62],
        [467.12, 677.77],
    ])

    pairs = [
        ("living_room", "IMG_3326.jpg", "IMG_3327.jpg", living_room_1, living_room_2),
        ("posters", "IMG_3331.jpg", "IMG_3332.jpg", posters_1, posters_2),
    ]
    for name, file_1, file_2, points_1, points_2 in pairs:
        save_pair(name, root / "previews" / file_1, root / "previews" / file_2, points_1, points_2, output_dir)

def mark_corners(image, corners):
    marked = image.copy()
    polygon = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(marked, [polygon], True, (40, 180, 60), 2, cv2.LINE_AA)
    for index, corner in enumerate(np.round(corners).astype(int), start=1):
        cv2.circle(marked, tuple(corner), 5, (40, 80, 220), -1, cv2.LINE_AA)
        cv2.putText(marked, str(index), tuple(corner + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return marked

def object_crop(warped, minimum, rectangle):
    origin_x = int(-minimum[0])
    origin_y = int(-minimum[1])
    width = int(rectangle[1, 0]) + 1
    height = int(rectangle[2, 1]) + 1
    return warped[origin_y:origin_y + height, origin_x:origin_x + width]

def save_rectification(name, image_path, corners, output_dir):
    image = cv2.imread(str(image_path))
    rectangle = rectangle_from_corners(corners)
    homography = computeH(corners, rectangle)
    cv2.imwrite(str(output_dir / f"{name}_corners.jpg"), mark_corners(image, corners))

    started = time.perf_counter()
    nearest, minimum = warp_image(image, homography, sample_nearest)
    nearest_seconds = time.perf_counter() - started
    started = time.perf_counter()
    bilinear, _ = warp_image(image, homography, sample_bilinear)
    bilinear_seconds = time.perf_counter() - started

    nearest_object = composite_warp(object_crop(nearest, minimum, rectangle))
    bilinear_object = composite_warp(object_crop(bilinear, minimum, rectangle))
    cv2.imwrite(str(output_dir / f"{name}_nearest.jpg"), nearest_object)
    cv2.imwrite(str(output_dir / f"{name}_bilinear.jpg"), bilinear_object)

    full_nearest = composite_warp(nearest)
    scale = 900 / max(full_nearest.shape[:2])
    if scale < 1:
        full_nearest = cv2.resize(full_nearest, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_dir / f"{name}_full_nearest.jpg"), full_nearest)

    patch_height = min(90, nearest_object.shape[0])
    patch_width = min(110, nearest_object.shape[1])
    nearest_patch = nearest_object[:patch_height, -patch_width:]
    bilinear_patch = bilinear_object[:patch_height, -patch_width:]
    zoom = np.hstack((
        cv2.resize(nearest_patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
        cv2.resize(bilinear_patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
    ))
    cv2.imwrite(str(output_dir / f"{name}_zoom.jpg"), zoom)
    print(name, f"nearest {nearest_seconds:.3f}s", f"bilinear {bilinear_seconds:.3f}s", "object", nearest_object.shape[1], nearest_object.shape[0])

def part_a_3():
    root = Path(__file__).resolve().parent
    output_dir = root / "a3"
    output_dir.mkdir(exist_ok=True)
    print_corners = np.array([
        [856.82, 270.44],
        [1081.60, 240.50],
        [1087.45, 651.69],
        [870.33, 636.27],
    ])
    window_corners = np.array([
        [373.70, 529.99],
        [1161.31, 528.50],
        [1154.80, 699.57],
        [378.36, 702.65],
    ])
    save_rectification("print", root / "previews" / "IMG_3331.jpg", print_corners, output_dir)
    save_rectification("window", root / "previews" / "IMG_3326.jpg", window_corners, output_dir)

def feather_weights(height, width):
    y = np.arange(height)
    x = np.arange(width)
    y_distance = np.minimum(y, height - 1 - y)
    x_distance = np.minimum(x, width - 1 - x)
    weights = np.minimum(y_distance[:, None], x_distance[None, :]).astype(np.float32)
    return weights / np.max(weights)

def bilinear_lookup(image, source_x, source_y):
    height, width = image.shape[:2]
    plane = image[:, :, None] if image.ndim == 2 else image
    plane = plane.astype(np.float32)
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    dx = (source_x - x0).astype(np.float32)
    dy = (source_y - y0).astype(np.float32)
    weight_00 = (1 - dx) * (1 - dy)
    weight_10 = dx * (1 - dy)
    weight_01 = (1 - dx) * dy
    weight_11 = dx * dy
    sampled = np.zeros(source_x.shape + (plane.shape[2],), dtype=np.float32)
    sampled[valid] = (
        plane[y0[valid], x0[valid]] * weight_00[valid, None]
        + plane[y0[valid], x1[valid]] * weight_10[valid, None]
        + plane[y1[valid], x0[valid]] * weight_01[valid, None]
        + plane[y1[valid], x1[valid]] * weight_11[valid, None]
    )
    if image.ndim == 2:
        sampled = sampled[:, :, 0]
    return sampled, valid

def warp_feather(image, homography, minimum, maximum):
    source_x, source_y = inverse_coordinates(minimum, maximum, homography)
    colors, valid = bilinear_lookup(image, source_x, source_y)
    weights, _ = bilinear_lookup(feather_weights(image.shape[0], image.shape[1]), source_x, source_y)
    weights *= valid
    return colors, weights

def mosaic_bounds(images, homographies):
    minima, maxima = [], []
    for image, homography in zip(images, homographies):
        minimum, maximum = warp_bounds(image, homography)
        minima.append(minimum)
        maxima.append(maximum)
    return np.min(minima, axis=0), np.max(maxima, axis=0)

def blend_feather(layers):
    total_color = np.zeros_like(layers[0][0])
    total_weight = np.zeros(layers[0][0].shape[:2], dtype=np.float32)
    for colors, weights in layers:
        total_color += colors * weights[:, :, None]
        total_weight += weights
    covered = total_weight > 0
    mosaic = np.full_like(total_color, 244)
    mosaic[covered] = total_color[covered] / total_weight[covered, None]
    return np.clip(np.round(mosaic), 0, 255).astype(np.uint8), covered

def blend_overwrite(layers):
    canvas = np.full_like(layers[0][0], 244)
    for colors, weights in layers:
        mask = weights > 0
        canvas[mask] = colors[mask]
    return np.clip(np.round(canvas), 0, 255).astype(np.uint8)

def crop_covered(image, covered):
    rows, cols = np.where(covered)
    return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]

def fit_width(image, width):
    if image.shape[1] <= width:
        return image
    scale = width / image.shape[1]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def save_mosaic(name, image_paths, homographies, output_dir):
    images = [cv2.imread(str(path)) for path in image_paths]
    minimum, maximum = mosaic_bounds(images, homographies)
    layers = [warp_feather(image, homography, minimum, maximum) for image, homography in zip(images, homographies)]
    feathered, covered = blend_feather(layers)
    overwritten = blend_overwrite(layers)
    cv2.imwrite(str(output_dir / f"{name}_feather.jpg"), fit_width(crop_covered(feathered, covered), 1800), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(output_dir / f"{name}_overwrite.jpg"), fit_width(crop_covered(overwritten, covered), 1800), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(name, "canvas", int(maximum[0] - minimum[0] + 1), int(maximum[1] - minimum[1] + 1))

def part_a_4():
    root = Path(__file__).resolve().parent
    output_dir = root / "a4"
    output_dir.mkdir(exist_ok=True)
    previews = root / "previews"
    living = [
        previews / "IMG_3325.jpg",
        previews / "IMG_3326.jpg",
        previews / "IMG_3327.jpg",
    ]
    living_homographies = [
        np.array([
            [1.76061638, -0.057863015, -581.913609],
            [0.327715021, 1.40160316, -259.040362],
            [0.00048624553, -0.0000380867767, 1.0],
        ]),
        np.eye(3),
        np.array([
            [0.479980822, 0.0355719592, 408.739925],
            [-0.206729392, 0.810182093, 112.843899],
            [-0.000333622026, 0.000028453448, 1.0],
        ]),
    ]
    posters = [
        previews / "IMG_3330.jpg",
        previews / "IMG_3331.jpg",
        previews / "IMG_3332.jpg",
    ]
    posters_homographies = [
        np.array([
            [1.71677249, 0.0969235426, -910.691798],
            [0.177501626, 1.47322113, -218.227445],
            [0.000425030021, 0.0000245012997, 1.0],
        ]),
        np.eye(3),
        np.array([
            [0.507528185, 0.0261236694, 628.319888],
            [-0.159974799, 0.871851115, 43.9517477],
            [-0.000314267788, 0.0000440848895, 1.0],
        ]),
    ]
    door = [
        previews / "IMG_3332.jpg",
        previews / "IMG_3333.jpg",
        previews / "IMG_3334.jpg",
    ]
    door_homographies = [
        np.array([
            [1.59492220, 0.00680062892, -758.469124],
            [0.200841773, 1.35673273, -173.136761],
            [0.000368992690, -0.0000165427839, 1.0],
        ]),
        np.eye(3),
        np.array([
            [0.447348873, 0.00472172883, 689.915498],
            [-0.199111074, 0.825776600, 90.4298818],
            [-0.000338917088, 0.00000440057977, 1.0],
        ]),
    ]
    save_mosaic("living", living, living_homographies, output_dir)
    save_mosaic("posters", posters, posters_homographies, output_dir)
    save_mosaic("door", door, door_homographies, output_dir)

def main():
    part_a_4()

if __name__ == "__main__":
    main()
