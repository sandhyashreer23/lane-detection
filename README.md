# lane-detection
Lane detection on real driving images using OpenCV and the KITTI Road Dataset — part of my ADAS learning journey.

 Lane Detection using OpenCV — KITTI Dataset

A computer vision project that detects lane markings on roads from real driving images. Built using classical image processing techniques as part of my journey into ADAS (Advanced Driver Assistance Systems).

What This Does

The system takes raw road images and draws the detected lane lines on them. It processes both the training and testing splits of the KITTI Road dataset automatically.

Here's the rough idea of how it works:

Raw Image → Grayscale → Blur → Edge Detection → Road Masking → Line Detection → Output

Each step has a specific reason behind it — grayscale removes unnecessary color data, blur reduces noise, Canny finds sharp edges where lane markings are, and Hough transform converts those edges into actual lines.

## Dataset

I used the **KITTI Road Dataset** — a benchmark dataset collected from a car driving through German streets, widely used in autonomous driving research.

- 289 training images
- 290 testing images
- Download: [www.cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti) → Road section

## Tech Stack

- Python 3.x
- OpenCV
- NumPy


## How to Run

1. Clone the repo
```bash
git clone https://github.com/yourusername/lane-detection
cd lane-detection
```

2. Install dependencies
```bash
pip install opencv-python numpy
```

3. Download the KITTI Road dataset and update the path in the script
```python
DATA_ROOT = r"path\to\your\data_road"
```

4. Run
```bash
python lane_detection.py
```

The script will run on both training and testing sets and save outputs to a `lane_detection_output` folder.


## Results

The model successfully detects left and right lane boundaries on straight roads. It handles varying lighting conditions reasonably well since it relies on edge gradients rather than color.

It struggles a bit on sharp curves — that's a known limitation of using straight-line Hough transforms. Polynomial fitting would handle curves better, which I plan to add next.



## What I Learned

- How the Canny edge detector actually works under the hood (gradient magnitude + non-maximum suppression)
- Why region masking matters — without it, the sky and trees create dozens of false positives
- The difference between training and testing evaluation on a real benchmark dataset
- How KITTI is structured and why it's used as a standard in the automotive industry

## What's Next

- [ ] Curve detection using polynomial fitting
- [ ] Lane departure warning overlay
- [ ] Test on night/low-light images using HLS color space
- [ ] Compare results with a deep learning approach (SCNN or LaneNet)

## References

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti)
- [OpenCV Hough Line Transform docs](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
- [Canny Edge Detection — original paper by John Canny, 1986]
