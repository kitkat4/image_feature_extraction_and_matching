#include <opencv2/opencv.hpp>

#include <my_utils_kk4.hpp>

#include <string>
#include <vector>
#include <iostream>

void readAndResizeImage(const std::string& image_path,
                        cv::Mat * const out_image,
                        const int max_rows,
                        const bool verbose){

    *out_image = cv::imread(image_path);
    if(out_image->empty()){
        std::cerr << "[ERROR] Could not open " << image_path << std::endl;
        return;
    }

    if(out_image->rows > max_rows){
        cv::resize(*out_image, *out_image, cv::Size(),
                   static_cast<double>(max_rows) / out_image->rows,
                   static_cast<double>(max_rows) / out_image->rows,
                   cv::INTER_LINEAR);
        if(verbose){
            std::cerr << "[ INFO] Successfully read " << image_path
                      << " (resized to " << out_image->cols << " x "
                      << out_image->rows << ")" << std::endl;
        }
    }else{
        if(verbose){
            std::cerr << "[ INFO] Successfully read " << image_path
                      << " (size: " << out_image->cols << " x "
                      << out_image->rows << ")" << std::endl;
        }
    }
}

int main(int argc, char** argv){

    if(argc != 3){
        std::cerr << "Usage: ./test0 <path to image 1> <path to image 2>"
                  << std::endl;
        return 1;
    }

    std::string img1_path(argv[1]);
    std::string img2_path(argv[2]);

    cv::Mat img1, img2;

    const int max_rows = 800;

    readAndResizeImage(img1_path, &img1, max_rows, true);
    readAndResizeImage(img2_path, &img2, max_rows, true);

    my_utils_kk4::StopWatch stop_watch;

    const int nfeatures = 500;
    const float scaleFactor = 1.2f;
    const int nlevels = 8;
    const int edgeThreshold = 31;
    const int firstLevel = 0;
    const int WTA_K = 2;
    const int scoreType = cv::ORB::HARRIS_SCORE;
    const int patchSize = 31;
    const int fastThreshold = 20;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels,
                                           edgeThreshold, firstLevel, WTA_K,
                                           scoreType, patchSize,
                                           fastThreshold);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    stop_watch.start();
    
    orb->detect(img1, keypoints1);

    std::cerr << "[DEBUG] " << __FILE__ << "(line " << __LINE__ << "): "
              << stop_watch.interval() * 1000 << " [ms]" << std::endl;;
        
    orb->detect(img2, keypoints2);

    std::cerr << "[DEBUG] " << __FILE__ << "(line " << __LINE__ << "): "
              << stop_watch.interval() * 1000 << " [ms]" << std::endl;;
    

    orb->compute(img1, keypoints1, descriptors1);

    std::cerr << "[DEBUG] " << __FILE__ << "(line " << __LINE__ << "): "
              << stop_watch.interval() * 1000 << " [ms]" << std::endl;;
    
    orb->compute(img2, keypoints2, descriptors2);
    
    std::cerr << "[DEBUG] " << __FILE__ << "(line " << __LINE__ << "): "
              << stop_watch.interval() * 1000 << " [ms]" << std::endl;;

    cv::Mat img_keypoints1, img_keypoints2;

    cv::drawKeypoints(img1, keypoints1, img_keypoints1, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DEFAULT);

    cv::drawKeypoints(img2, keypoints2, img_keypoints2, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DEFAULT);

    const std::string window_name("test0");

    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    cv::imshow(window_name, img_keypoints1);

    // !!! CAUTION !!!
    // These parameters may depend on what kind of feature point you use!
    const int normType = cv::NORM_HAMMING;
    const bool crossCheck = true;

    // Brute-force descriptor matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(normType, crossCheck);

    std::vector<cv::DMatch> matches;

    std::cerr << "[DEBUG] " << __FILE__ << "(line " << __LINE__ << "): "
              << stop_watch.interval() * 1000 << " [ms]" << std::endl;;

    matcher->match(descriptors1, descriptors2, matches);

    std::cerr << "[DEBUG] " << __FILE__ << "(line " << __LINE__ << "): "
              << stop_watch.interval() * 1000 << " [ms]" << std::endl;;

    cv::Mat img_matches;

    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches,
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                    std::vector<char>(), cv::DrawMatchesFlags::DEFAULT);

    cv::imshow(window_name, img_matches);

    cv::waitKey(0);

    cv::destroyAllWindows();

    return 0;
}


