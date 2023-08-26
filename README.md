# KeypointProfiler
# Data Analysis: Key Point Detection Algorithms

This README provides an analysis of the data related to various key point detection algorithms. The data consists of measurements obtained from two different approaches: a basic sequential method and a parallel method where the image was divided into four patches and processed concurrently by separate threads. The measurements include the number of keypoints detected and the duration (in milliseconds) for each algorithm in both full-sized and half-sized image.

## Dataset

The dataset contains the following information for each algorithm:

- **Method**: The name of the key point detection algorithm along with the execution context (CPU or GPU) and the processing mode (Parallel or Sequential).
- **Number of Keypoints Detected**: The total number of keypoints detected by the algorithm.
- **Duration (Milliseconds)**: The time taken by the algorithm to execute and detect keypoints in a full-sized image.
- **Number of Keypoints Detected in half-sized image**: The number of keypoints detected by the algorithm when applied to a half-sized image.
- **Duration (Milliseconds) in half-sized image**: The time taken by the algorithm to execute and detect keypoints in a half-sized image.

## Analysis

Based on the provided data, here are some key observations:

1. **Number of Keypoints Detected**:

   - Among the parallel algorithms, SIFT (CPU) detected the highest number of keypoints in both the full-sized and half-sized images, with 319,333 and 51,474 keypoints, respectively.
   - AKAZE (CPU) and AKAZE (GPU) detected the same number of keypoints, with 87,064 keypoints in both the full-sized and half-sized images.
   - ORB (CPU) and ORB (GPU) detected the lowest number of keypoints, with 8,000 keypoints in both the full-sized and half-sized images.

1. **Duration**:

   - For the full-sized images, SIFT (CPU) took the longest time to execute, with a duration of 3,541 milliseconds. It was followed by SIFT (GPU) with 467 milliseconds.
   - ORB (GPU) had the shortest execution time, taking only 131 milliseconds to complete.
   - In the half-sized images, the execution times decreased for all algorithms compared to the full-sized images.
