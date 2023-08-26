#ifndef FEATURE_PROCESSORS_HPP
#define FEATURE_PROCESSORS_HPP

#include <opencv2/opencv.hpp>

class FeatureProcessor {
public:
    virtual int computeFeatures(const cv::Mat& image) = 0;
    virtual ~FeatureProcessor() {}
};

class AKAZEProcessor : public FeatureProcessor {
public:
    int computeFeatures(const cv::Mat& image) override;
};

class SIFTProcessor : public FeatureProcessor {
public:
    int computeFeatures(const cv::Mat& image) override;
};

class ORBProcessor : public FeatureProcessor {
public:
    int computeFeatures(const cv::Mat& image) override;
};

#endif // FEATURE_PROCESSORS_HPP