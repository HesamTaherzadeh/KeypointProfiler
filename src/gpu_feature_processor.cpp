#include "../include/gpu_feature_processor.hpp"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

int gpuSIFTProcessor::computeFeatures(const cv::Mat& image) {
    cv::cuda::GpuMat d_image(image);
    cv::cuda::GpuMat d_descriptors;
    cv::cuda::GpuMat d_keypoints;

    cv::Ptr<cv::cuda::ORB> sift = cv::cuda::ORB::create();
    sift->detectAndComputeAsync(d_image, cv::noArray(), d_keypoints, d_descriptors);

    // Create a CUDA stream
    cv::cuda::Stream stream;

    // Synchronize the stream to ensure the computation is complete
    stream.waitForCompletion();

    std::vector<cv::KeyPoint> h_keypoints;
    sift->convert(d_keypoints, h_keypoints);

    return h_keypoints.size();
}

int gpuORBProcessor::computeFeatures(const cv::Mat& image) {
    cv::cuda::GpuMat d_image(image);
    cv::cuda::GpuMat d_descriptors;
    cv::cuda::GpuMat d_keypoints;

    cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();

    orb->detectAndComputeAsync(d_image, cv::noArray(), d_keypoints, d_descriptors);

    // Create a CUDA stream
    cv::cuda::Stream stream;

    // Synchronize the stream to ensure the computation is complete
    stream.waitForCompletion();

    std::vector<cv::KeyPoint> h_keypoints;
    orb->convert(d_keypoints, h_keypoints);

    return h_keypoints.size();
}