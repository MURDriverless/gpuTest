#include <iostream>
#include <chrono>
#include <vector>

#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

const unsigned long RUN_SIZE = 100;
char INTERP_STR[][10] = {"Nearest", "Linear", "Cubic"};
cv::InterpolationFlags INTERP_METHOD = cv::InterpolationFlags::INTER_CUBIC;

char MAPPING_STR[][15] = {"NONE", "PAGE_LOCKED", "SHARED", "WRITE_COMBINED"};
enum HOSTMAPPING {
    NONE = 0,
    PAGE_LOCKED = 1,
    SHARED = 2,
    WRITE_COMBINED = 3
};

cv::Mat cameraMatrix, distCoeffs;
cv::Size calibSize;

cv::Mat xmap, ymap;
cv::cuda::GpuMat xmap_gpu, ymap_gpu;

unsigned long VectMean(std::vector<unsigned long> &timeVect);
void TestCpu(std::vector<unsigned long> &timeVect);
void TestGpu(std::vector<unsigned long> &timeVect);
void TestMapped(std::vector<unsigned long> &timeVect, cv::cuda::HostMem::AllocType allocType);
void TestMapped_Shared(std::vector<unsigned long> &timeVect);

int main(int argc, char** argv) {
    int hostMemMode = 0;
    if (argc > 1) {
        int modeIdx = atoi(argv[1]);

        hostMemMode = (modeIdx >= 0 && modeIdx <= 3) ? modeIdx : 0;
    }
    
    std::cout << "Interp Method: " << INTERP_STR[INTERP_METHOD] << std::endl;
    std::cout << "GPU HostMem alloc type: " << MAPPING_STR[hostMemMode] << std::endl;
    std::cout << "Num runs: " << RUN_SIZE << std::endl;


    std::cout << std::endl;

    cv::FileStorage fs;
    fs.open("../calibration.xml", cv::FileStorage::READ);
    fs["cameraMatrix"]   >> cameraMatrix;
    fs["distCoeffs"]     >> distCoeffs;
    fs["calibImageSize"] >> calibSize;

    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, calibSize, 0.0);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, cv::Size(1920, 1200), CV_32FC1, xmap, ymap);
    xmap_gpu.upload(xmap);
    ymap_gpu.upload(ymap);

    std::vector<unsigned long> cpuTimeVect;
    TestCpu(cpuTimeVect);

    std::vector<unsigned long> gpuTimeVect;

    switch (hostMemMode) {
        case PAGE_LOCKED:
            TestMapped(gpuTimeVect, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
            break;
    
        case SHARED:
            TestMapped_Shared(gpuTimeVect);
            break;

        case WRITE_COMBINED:
            TestMapped(gpuTimeVect, cv::cuda::HostMem::AllocType::WRITE_COMBINED);
            break;

        default:
            TestGpu(gpuTimeVect);
            break;
    }

    std::cout << "CPU mean time taken: " << VectMean(cpuTimeVect) << std::endl;
    std::cout << "GPU mean time taken: " << VectMean(gpuTimeVect) << std::endl;

    return 0;
}

unsigned long VectMean(std::vector<unsigned long> &timeVect) {
    if (timeVect.size() == 0) {
        return 0;
    }

    unsigned long mean = 0;

    for (const auto& time : timeVect) {
        mean += time;
    }

    return mean/timeVect.size();
}

void TestCpu(std::vector<unsigned long> &timeVect) {
    cv::Mat testImage(calibSize, CV_8UC1);
    cv::Mat testImageColor(calibSize, CV_8UC3);
    cv::Mat testImageColor_Undist(calibSize, CV_8UC3);

    for (int i = 0; i < RUN_SIZE; i++) {
        cv::randu(testImage, 0, 255);

        auto then = std::chrono::high_resolution_clock::now();

        cv::cvtColor(testImage, testImageColor, cv::COLOR_BayerBG2BGR);
        cv::remap(testImageColor, testImageColor_Undist, xmap, ymap, INTERP_METHOD);

        auto now = std::chrono::high_resolution_clock::now();
        unsigned long deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
        timeVect.push_back(deltaT);
    }
}

void TestGpu(std::vector<unsigned long> &timeVect) {
    cv::Mat testImage(calibSize, CV_8UC1);
    cv::Mat testImageColor_Undist(calibSize, CV_8UC3);

    cv::cuda::GpuMat testImage_gpu;
    cv::cuda::GpuMat testImageColor_gpu;
    cv::cuda::GpuMat testImageColor_Undist_gpu;

    for (int i = 0; i < RUN_SIZE; i++) {
        cv::randu(testImage, 0, 255);

        auto then = std::chrono::high_resolution_clock::now();

        testImage_gpu.upload(testImage);
        cv::cuda::cvtColor(testImage_gpu, testImageColor_gpu, cv::COLOR_BayerBG2BGR);
        cv::cuda::remap(testImageColor_gpu, testImageColor_Undist_gpu, xmap_gpu, ymap_gpu, INTERP_METHOD);
        testImageColor_Undist_gpu.download(testImageColor_Undist);

        auto now = std::chrono::high_resolution_clock::now();
        unsigned long deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
        timeVect.push_back(deltaT);
    }
}

void TestMapped(std::vector<unsigned long> &timeVect, cv::cuda::HostMem::AllocType allocType) {
    cv::cuda::HostMem testImage(calibSize, CV_8UC1, allocType);
    cv::cuda::HostMem testImageColor_Undist(calibSize, CV_8UC3, allocType);

    cv::cuda::GpuMat testImage_gpu;
    cv::cuda::GpuMat testImageColor_gpu;
    cv::cuda::GpuMat testImageColor_Undist_gpu;

    for (int i = 0; i < RUN_SIZE; i++) {
        cv::randu(testImage, 0, 255);

        auto then = std::chrono::high_resolution_clock::now();

        testImage_gpu.upload(testImage);
        cv::cuda::cvtColor(testImage_gpu, testImageColor_gpu, cv::COLOR_BayerBG2BGR);
        cv::cuda::remap(testImageColor_gpu, testImageColor_Undist_gpu, xmap_gpu, ymap_gpu, INTERP_METHOD);
        testImageColor_Undist_gpu.download(testImageColor_Undist);

        auto now = std::chrono::high_resolution_clock::now();
        unsigned long deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
        timeVect.push_back(deltaT);
    }
}

void TestMapped_Shared(std::vector<unsigned long> &timeVect) {
    cv::cuda::HostMem testImage(calibSize, CV_8UC1, cv::cuda::HostMem::SHARED);
    cv::cuda::HostMem testImageColor_Undist(calibSize, CV_8UC3, cv::cuda::HostMem::SHARED);

    cv::cuda::GpuMat testImage_gpu = testImage.createGpuMatHeader();
    cv::cuda::GpuMat testImageColor_gpu;
    cv::cuda::GpuMat testImageColor_Undist_gpu = testImageColor_Undist.createGpuMatHeader();

    for (int i = 0; i < RUN_SIZE; i++) {
        cv::randu(testImage, 0, 255);

        auto then = std::chrono::high_resolution_clock::now();

        testImage_gpu.upload(testImage);
        cv::cuda::cvtColor(testImage_gpu, testImageColor_gpu, cv::COLOR_BayerBG2BGR);
        cv::cuda::remap(testImageColor_gpu, testImageColor_Undist_gpu, xmap_gpu, ymap_gpu, INTERP_METHOD);
        testImageColor_Undist_gpu.download(testImageColor_Undist);

        auto now = std::chrono::high_resolution_clock::now();
        unsigned long deltaT = std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count();
        timeVect.push_back(deltaT);
    }
}
