#include "../include/feature_processors.hpp"

int AKAZEProcessor::computeFeatures(const cv::Mat& image) {
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    return keypoints.size();
}

int SIFTProcessor::computeFeatures(const cv::Mat& image) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    return keypoints.size();
}

int ORBProcessor::computeFeatures(const cv::Mat& image) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    return keypoints.size();
}