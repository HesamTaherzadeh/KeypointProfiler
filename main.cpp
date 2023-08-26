#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "include/feature_processors.hpp"
#include "include/gpu_feature_processor.hpp"

void divideImageIntoPatches(const cv::Mat& image, int numPatches, std::vector<cv::Mat>& imagePatches) {
    int patchWidth = image.cols / numPatches;
    int patchHeight = image.rows / numPatches;

    for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numPatches; j++) {
            int startX = i * patchWidth;
            int startY = j * patchHeight;
            cv::Rect patchROI(startX, startY, patchWidth, patchHeight);
            cv::Mat patch = image(patchROI).clone();
            imagePatches.push_back(patch);
        }
    }
}

class Timer {
public:
    Timer(const std::string& blockName)
        : blockName_(blockName), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << blockName_ << " took " << duration << " milliseconds." << std::endl;
    }

private:
    std::string blockName_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class ImageProcessor {
public:
    int processImage(const cv::Mat& image, FeatureProcessor& featureProcessor, int numPatches, bool parallel = true) {
        int totalKeypoints = 0;

        if (parallel) {
            totalKeypoints = processImageParallel(image, featureProcessor, numPatches);
        } else {
            totalKeypoints = processImageSequential(image, featureProcessor, numPatches);
        }

        return totalKeypoints;
    }

private:
    int processImageSequential(const cv::Mat& image, FeatureProcessor& featureProcessor, int numPatches) {
        std::vector<cv::Mat> imagePatches;
        divideImageIntoPatches(image, numPatches, imagePatches);

        int totalKeypoints = 0;

        for (const auto& patch : imagePatches) {
            totalKeypoints += featureProcessor.computeFeatures(patch);
        }

        return totalKeypoints;
    }

    int processImageParallel(const cv::Mat& image, FeatureProcessor& featureProcessor, int numPatches) {
        std::vector<cv::Mat> imagePatches;
        divideImageIntoPatches(image, numPatches, imagePatches);

        int totalKeypoints = 0;

        std::vector<std::thread> threads;
        std::mutex mutex;

        for (const auto& patch : imagePatches) {
            threads.emplace_back([&]() {
                int keypoints = featureProcessor.computeFeatures(patch);
                std::lock_guard<std::mutex> lock(mutex);
                totalKeypoints += keypoints;
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        return totalKeypoints;
    }
};

void computeFeaturesWithTimer(const std::string& processorName, ImageProcessor& processor,
                              FeatureProcessor& featureProcessor, const cv::Mat& image, int numPatches, bool parallel) {
    Timer timer("Compute Features (" + processorName + (parallel ? " - Parallel)" : " - Sequential)"));
    int numKeypoints = processor.processImage(image, featureProcessor, numPatches, parallel);
    std::cout << processorName << (parallel ? " - Parallel: " : " - Sequential: ") << "Number of keypoints detected = " << numKeypoints << std::endl;
}

// void computeFeaturesWithTimer(const std::string& processorName, ImageProcessor& processor,
//                               FeatureProcessor& featureProcessor, const cv::Mat& image, int numPatches, bool parallel) {
//     Timer timer("Compute Features (" + processorName + (parallel ? " - Parallel)" : " - Sequential)"));
//     int numKeypoints = processor.processImage(image, featureProcessor, numPatches, parallel);
//     std::cout << processorName << (parallel ? " - Parallel: " : " - Sequential: ") << "Number of keypoints detected = " << numKeypoints << std::endl;
// }

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./program <image_path> <width> <height>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);

    cv::Mat image_not_resized = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image_not_resized.empty()) {
        std::cout << "Failed to read the image." << std::endl;
        return -1;
    }
    cv::Mat image;
    cv::resize(image_not_resized, image, cv::Size(width, height));
    int numPatches = 4;
    ImageProcessor image_processor;
    std::unique_ptr<FeatureProcessor> akazeProcessor = std::make_unique<AKAZEProcessor>();
    std::unique_ptr<FeatureProcessor> siftProcessor = std::make_unique<SIFTProcessor>();
    std::unique_ptr<FeatureProcessor> orbProcessor = std::make_unique<ORBProcessor>();
    std::unique_ptr<GPUFeatureProcessor> gpusiftProcessor = std::make_unique<gpuSIFTProcessor>();
    std::unique_ptr<GPUFeatureProcessor> gpuorbProcessor = std::make_unique<gpuORBProcessor>();

    computeFeaturesWithTimer("AKAZE - CPU", image_processor, *akazeProcessor, image, numPatches, true);
    computeFeaturesWithTimer("AKAZE - CPU", image_processor, *akazeProcessor, image, numPatches, false);
    computeFeaturesWithTimer("SIFT - CPU", image_processor, *siftProcessor, image, numPatches, true);
    computeFeaturesWithTimer("SIFT - CPU", image_processor, *siftProcessor, image, numPatches, false);
    computeFeaturesWithTimer("ORB - CPU", image_processor, *orbProcessor, image, numPatches, true);
    computeFeaturesWithTimer("ORB - CPU", image_processor, *orbProcessor, image, numPatches, false);
    computeFeaturesWithTimer("SIFT - GPU", image_processor, *gpusiftProcessor, image, numPatches, true);
    computeFeaturesWithTimer("SIFT - GPU", image_processor, *gpusiftProcessor, image, numPatches, false);
    computeFeaturesWithTimer("ORB - GPU", image_processor, *gpuorbProcessor, image, numPatches, true);
    computeFeaturesWithTimer("ORB - GPU", image_processor, *gpuorbProcessor, image, numPatches, false);

return 0;}