import cv2
import numpy as np
import os


# I'm defining the base path here so I don't have to change it in multiple places
DATA_ROOT   = r"C:\Users\sandhya shree R\Downloads\lane\input\data_road_right"
OUTPUT_ROOT = r"C:\Users\sandhya shree R\Downloads\lane\output"


def detect_lanes(frame):
    # First I convert to grayscale because color info isn't needed for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # A little blur helps remove noise so we don't pick up fake edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny finds the sharp edges in the image — lanes appear as strong edges
    edges = cv2.Canny(blurred, 50, 150)

    # I only want to look at the road area, not the sky or trees
    # So I mask everything outside this triangle
    h, w = edges.shape
    road_area = np.array([[(0, h), (w, h), (w // 2, int(h * 0.55))]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, road_area, 255)
    road_edges = cv2.bitwise_and(edges, mask)

    # Hough transform finds straight lines from the edges
    lines = cv2.HoughLinesP(road_edges, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)

    # Separate lines into left lane and right lane based on their slope
    left_points  = []
    right_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # skip vertical lines to avoid division by zero
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < -0.5:
                left_points.append((slope, intercept))
            elif slope > 0.5:
                right_points.append((slope, intercept))

    # Draw the averaged lane lines on a blank canvas then blend with original
    canvas = np.zeros_like(frame)

    for points in [left_points, right_points]:
        if not points:
            continue
        avg_slope, avg_intercept = np.mean(points, axis=0)
        if avg_slope == 0:
            continue
        y_bottom = h
        y_top    = int(h * 0.6)
        x_bottom = int((y_bottom - avg_intercept) / avg_slope)
        x_top    = int((y_top    - avg_intercept) / avg_slope)
        cv2.line(canvas, (x_bottom, y_bottom), (x_top, y_top), (0, 255, 0), 10)

    return cv2.addWeighted(frame, 0.8, canvas, 1, 1)


def process_folder(images_path, save_path, label):
    if not os.path.exists(images_path):
        print(f"  Skipping {label} — folder not found: {images_path}")
        return

    os.makedirs(save_path, exist_ok=True)
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".png")])

    print(f"\n  {label}  —  {len(image_files)} images found")
    print(f"  Press Q to skip to the next set\n")

    for i, filename in enumerate(image_files):
        img = cv2.imread(os.path.join(images_path, filename))
        if img is None:
            continue

        output = detect_lanes(img)
        cv2.imwrite(os.path.join(save_path, filename), output)
        cv2.imshow(f"Lane Detection  [{label}]", output)

        print(f"  {i+1}/{len(image_files)}   {filename}")

        if cv2.waitKey(200) & 0xFF == ord('q'):
            print("  Skipping remaining images...")
            break

    cv2.destroyAllWindows()
    print(f"\n  Done! Results saved to: {save_path}")


if __name__ == "__main__":
    print("\nLane Detection — KITTI Dataset")
    print(f"Data folder  : {DATA_ROOT}")
    print(f"Output folder: {OUTPUT_ROOT}")

    process_folder(
        images_path = os.path.join(DATA_ROOT,  "training", "image_3"),
        save_path   = os.path.join(OUTPUT_ROOT, "training_results"),
        label       = "TRAINING SET"
    )

    process_folder(
        images_path = os.path.join(DATA_ROOT,  "testing", "image_3"),
        save_path   = os.path.join(OUTPUT_ROOT, "testing_results"),
        label       = "TESTING SET"
    )

    print("\nAll done! Check your output folder.")