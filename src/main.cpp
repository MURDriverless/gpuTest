#include <iostream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

const unsigned long RUN_SIZE = 100;
char INTERP_STR[][10] = {"Nearest", "Linear", "Cubic"};
cv::InterpolationFlags INTERP_METHOD = cv::InterpolationFlags::INTER_LINEAR;

int main(int argc, char** argv) {
    std::cout << "Interp Method: " << INTERP_STR[INTERP_METHOD] << std::endl;
    std::cout << "Num runs: " << RUN_SIZE << std::endl;
    std::cout << std::endl;

    cv::Mat cameraMatrix, distCoeffs;
    cv::Size calibSize;

    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);
    fs["cameraMatrix"]   >> cameraMatrix;
    fs["distCoeffs"]     >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    cv::Mat xmap, ymap;
    cv::cuda::GpuMat xmap_gpu, ymap_gpu;

    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, calibSize, 0.0);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(1920, 1200), CV_32FC1, xmap, ymap);
    xmap_gpu.upload(xmap);
    ymap_gpu.upload(ymap);

    cv::Mat testImage(calibSize, CV_8UC1);
    cv::Mat testImageColor(calibSize, CV_8UC3);
    cv::Mat testImageColor_Undist(calibSize, CV_8UC3);

    cv::cuda::GpuMat testImage_gpu;
    cv::cuda::GpuMat testImageColor_gpu;
    cv::cuda::GpuMat testImageColor_Undist_gpu;

    unsigned long cpuMeanTime = 0;
    for (int i = 0; i < RUN_SIZE; i++) {
        cv::randu(testImage, 0, 255);

        auto then = std::chrono::high_resolution_clock::now();

        cv::cvtColor(testImage, testImageColor, cv::COLOR_BayerBG2BGR);
        cv::remap(testImageColor, testImageColor_Undist, xmap, ymap, INTERP_METHOD);

        auto now = std::chrono::high_resolution_clock::now();
        cpuMeanTime += std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
    }

    unsigned long gpuMeanTime = 0;
    for (int i = 0; i < RUN_SIZE; i++) {
        cv::randu(testImage, 0, 255);

        auto then = std::chrono::high_resolution_clock::now();

        testImage_gpu.upload(testImage);
        cv::cuda::cvtColor(testImage_gpu, testImageColor_gpu, cv::COLOR_BayerBG2BGR);
        cv::cuda::remap(testImageColor_gpu, testImageColor_Undist_gpu, xmap_gpu, ymap_gpu, INTERP_METHOD);
        testImageColor_Undist_gpu.download(testImageColor_Undist);

        auto now = std::chrono::high_resolution_clock::now();
        gpuMeanTime += std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
    }

    std::cout << "CPU mean time taken: " << cpuMeanTime/RUN_SIZE << std::endl;
    std::cout << "GPU mean time taken: " << gpuMeanTime/RUN_SIZE << std::endl;

    return 0;
}