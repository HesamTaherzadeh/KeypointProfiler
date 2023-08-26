#ifndef GPU_FEATURE_PROCESSOR_HPP
#define GPU_FEATURE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include "feature_processors.hpp"



class GPUFeatureProcessor: public FeatureProcessor {
public:
    virtual int computeFeatures(const cv::Mat& image) override = 0;
};

class gpuSIFTProcessor : public GPUFeatureProcessor {
public:
    int computeFeatures(const cv::Mat& image);
};

class gpuORBProcessor : public GPUFeatureProcessor {
public:
    int computeFeatures(const cv::Mat& image);
};

#endif /* GPU_FEATURE_PROCESSOR_HPP */