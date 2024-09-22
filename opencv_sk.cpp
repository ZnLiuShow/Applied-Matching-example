// opencv_sk.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <Windows.h>
#include "SimpleIni.h"
#include "../picturematch/PictureMatch.h"


void plusExm(char* srcpath, char* telpath);
void singleExm(char* srcpath, char* telpath);
void test(char* srcpath);
void test2(char* srcpath);
void test3(char* srcpath);
void test4(char* srcpath);
void test5(char* srcpath);
void test7();

int main(int argc, char** argv)
{ 
    test5(argv[1]);
    return 0;
    if (argc ==2)
    {
        test5(argv[1]);
        return -1;
    }
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [imagepath] argc:%d %s\n", argv[0], argc, argv[3]);
        return -1;
    }
    std::string telpath = argv[2];
    if (telpath.find('.') == std::string::npos)
    {
        plusExm(argv[1], argv[2]);
    }
    else
    {
        singleExm(argv[1], argv[2]);
    }
}

double getinidouble(const char* iniFile, const char* section) {
    std::cout << iniFile << std::endl;
    std::cout << section << std::endl;
    CSimpleIniA ini;
    auto rc= ini.LoadFile(iniFile);
    ini.SetUnicode(true);
    //std::cout << rc << std::endl;
    if (rc < 0)
        return 0.92;
    return std::stod(ini.GetValue(section, "p", "0.92"));
}
bool contains(std::string str1, std::string str2) {
    // 将两个字符串转换为小写
    transform(str1.begin(), str1.end(), str1.begin(), ::tolower);
    transform(str2.begin(), str2.end(), str2.begin(), ::tolower);

    // 判断第一个字符串是否包含第二个字符串
    return str1.find(str2) != std::string::npos;
}

std::vector<imginfo> getimgfile(std::string path) {
	//std::vector<std::string> t;
    std::vector<imginfo> r;
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		auto filename = entry.path().filename();
		if (std::filesystem::is_regular_file(entry.status())) {
            if (!contains(filename.string(), ".jpg") && !contains(filename.string(), ".png"))
                continue;

            r.push_back({ filename.string().substr(0, filename.string().find_last_of('.')),
                cv::imread(path + "\\" + filename.string()),
                getinidouble((path + "\\p.ini").c_str(),filename.string().substr(0, filename.string().find_last_of('.')).c_str())});
			//t.push_back(path + "\\" + filename.string());
			//std::cout << "Found file: " << path + "\\" + filename.string()<< "  "<< filename.string().substr(0, filename.string().find_last_of('.')) << std::endl;
		}
	}	
    return r;
}

void getimgfile(std::string path,std::vector<cv::Mat>&imgs,std::vector<std::string>&imgname,std::vector<double>& pcs) {
    imgs.clear();
    imgname.clear();
    pcs.clear();
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        auto filename = entry.path().filename();
        if (std::filesystem::is_regular_file(entry.status())) {
            if (!contains(filename.string(), ".jpg") && !contains(filename.string(), ".png"))
                continue;
            imgs.push_back(cv::imread(path + "\\" + filename.string()));
            imgname.push_back(filename.string().substr(0, filename.string().find_last_of('.')));
            pcs.push_back(getinidouble((path + "\\p.ini").c_str(), filename.string().substr(0, filename.string().find_last_of('.')).c_str()));
        }
    }
}
void plusExm(char* srcpath, char* telpath) {
    std::vector<cv::Mat> imgs;
    std::vector<std::string> imgname;
    std::vector<double> pcs;
    getimgfile(std::string(telpath), imgs, imgname, pcs);
    for (size_t i = 0; i < imgs.size(); i++)
    {
       // cv::imshow("out", imgs[i]);
        std::cout << "name: " << imgname[i] << " precision:" << pcs[i] << std::endl;
        //cv::waitKey(0);
    }

    cv::Mat img1 = cv::imread(srcpath);
    PictureMatch pm;
    double start = (double)cv::getTickCount();
    auto rs = pm.getmatchboxes(img1, imgs);
    std::cout << (((double)cv::getTickCount() - start)) / cv::getTickFrequency() << std::endl;
    for (size_t i = 0; i < rs.size(); i++)
    {
        if (rs[i].pixsc <= pcs[i])
            continue;
        std::cout << imgname[i] << "  设置精度：" << pcs[i] << "  编号:"<< i << " 检测精度："<< rs[i].pixsc << " 灰度：" << rs[i].gray<< " 坐标："<< rs[i].getp1() << ","<< rs[i].getp2() << std::endl;
        cv::rectangle(img1, rs[i].getp1(), rs[i].getp2(), cv::Scalar(0, 255, 0));//画出轮廓范围
        cv::putText(img1, rs[i].gray > 0.4 ?"cd": std::to_string(i), cv::Point(rs[i].getp1().x, rs[i].getp1().y + 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    }
    cv::imshow(srcpath, img1);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void singleExm(char* srcpath, char* telpath) {
    cv::Mat img1 = cv::imread(srcpath);
    cv::Mat img2 = cv::imread(telpath);
    PictureMatch pm;
    double start = (double)cv::getTickCount();
    MatchBox r = pm.getmatchbox(img1, img2);
    std::cout << (((double)cv::getTickCount() - start)) / cv::getTickFrequency() << std::endl;
    std::cout << "相似度1 " << r.score << std::endl;
    std::cout << "相似度2 " << r.pixsc << std::endl;
    cv::rectangle(img1, r.getp1(), r.getp2(), cv::Scalar(0, 255, 0));//画出轮廓范围

    cv::imshow(srcpath, img1);
    cv::imshow(telpath, img2);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void test(char* srcpath) {
    cv::Mat img1 = cv::imread(srcpath);
    cv::Mat grayImage;
    // 将彩色图片转换为灰度图片
    cv::cvtColor(img1, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat binaryImage = cv::Mat::zeros(grayImage.size(), grayImage.type());
    cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat threeChannelImage;
    std::vector<cv::Mat> channels = { binaryImage,  binaryImage,  binaryImage };
    cv::merge(channels, threeChannelImage);

    //PictureMatch pm;
    //double start = (double)cv::getTickCount();
    //auto r = pm.dealcolor(img1, cv::Scalar(146, 197, 214),15);
    //std::cout << (((double)cv::getTickCount() - start)) / cv::getTickFrequency() << std::endl;
    //cv::rectangle(img1, r.getp1(), r.getp2(), cv::Scalar(0, 255, 0));//画出轮廓范围

    cv::imshow(srcpath, threeChannelImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void test2(char* srcpath) {
    cv::Mat img = cv::imread(srcpath);
    cv::Mat tired = img(cv::Rect(655, 592, 134, 6)).clone();
    cv::rectangle(img, cv::Rect(655, 592, 134, 6), cv::Scalar(0, 128, 0));//draw rectangle
    PictureMatch pm;
    std::cout << pm.blueper(tired) << std::endl;
    std::cout << pm.redper(tired) << std::endl;
    cv::imshow(srcpath, img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void test3(char* srcpath) {
    // 读取图片
    cv::Mat src = cv::imread(srcpath);
    if (src.empty()) {
        std::cout << "图片加载失败" << std::endl;
    }

    // 将图片从BGR颜色空间转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // 设定白色宋体的HSV范围
    // 这里需要根据实际的白色描绘进行调整
    // 例如: 白色可能对应 hue 范围为 [0,180], saturation 范围为 [0.3,1], value 范围为 [0.8,1]
    cv::Scalar whiteLower(0, 0, 250); // HSV中白色的下界
    cv::Scalar whiteUpper(1, 1, 255); // HSV中白色的上界

    // 根据HSV值构造掩膜
    cv::Mat mask;
    cv::inRange(hsv, whiteLower, whiteUpper, mask);

    // 对掩膜进行膨胀和腐蚀，去除噪声
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    //cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1));

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓
    cv::Mat dst = src.clone();
    cv::drawContours(dst, contours, -1, cv::Scalar(0, 0, 255), 2);

    for (size_t i = 0; i < contours.size(); i++) {
        cv::RotatedRect rect = cv::minAreaRect(contours[i]); // 获取最小面积矩形
        cv::Point2f vertices[4];
        rect.points(vertices); // 获取矩形的四个顶点

        for (int j = 0; j < 4; j++) {
            line(dst, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2); // 在原图上画矩形
        }
    }

    // 显示结果
    cv::imshow("Original", src);
    cv::imshow("Mask", mask);
    cv::imshow("Contours", dst);
    cv::waitKey(0);
}

void test4(char* srcpath) {
    // 读取图片
    cv::Mat src = cv::imread(srcpath);
    if (src.empty()) {
        std::cout << "图片加载失败" << std::endl;
    }

    // 将图片从BGR颜色空间转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // 设定白色宋体的HSV范围
    // 这里需要根据实际的白色描绘进行调整
    // 例如: 白色可能对应 hue 范围为 [0,180], saturation 范围为 [0.3,1], value 范围为 [0.8,1]
    cv::Scalar whiteLower(0, 0, 250); // HSV中白色的下界
    cv::Scalar whiteUpper(1, 1, 255); // HSV中白色的上界

    // 根据HSV值构造掩膜
    cv::Mat mask;
    cv::inRange(hsv, whiteLower, whiteUpper, mask);

    // 对掩膜进行膨胀和腐蚀，去除噪声
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1));
    //cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1));

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓
    cv::Mat dst = src.clone();
    cv::drawContours(dst, contours, -1, cv::Scalar(0, 0, 255), 2);

    for (size_t i = 0; i < contours.size(); i++) {
        // 计算轮廓的边界矩形
        cv::Rect rect = cv::boundingRect(contours[i]);
        // 在原图像上绘制矩形
        cv::rectangle(dst, rect, cv::Scalar(255, 0, 0), 2);
    }

    // 显示结果
    cv::imshow("Original", src);
    cv::imshow("Mask", mask);
    cv::imshow("Contours", dst);
    cv::waitKey(0);
}

void test5(char* srcpath) {
    // 读取图片，假设图片是黑色像素包围的轮廓
    cv::Mat img = cv::imread(srcpath, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(srcpath);
    // 阈值化处理，将非黑色的像素变为白色
    cv::Mat thresh;
    cv::threshold(img, thresh, 74, 255, cv::THRESH_BINARY_INV);
    cv::imshow("thresh", thresh);
    cv::imshow("img", img);
    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓
    //cv::Mat contourImg = cv::Mat::zeros(img.size(), CV_8UC3);
    //for (size_t i = 0; i < contours.size(); i++) {
    //    cv::Scalar color = cv::Scalar(0, 255, 0); // 绿色轮廓
    //    cv::drawContours(img2, contours, static_cast<int>(i), color, 1);
    //}

    std::vector<cv::Mat> mtxts;
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < 14)
            continue;
        // 计算轮廓的边界矩形
        cv::Rect rect = cv::boundingRect(contours[i]);
        // 在原图像上绘制矩形
        mtxts.push_back(img2(rect).clone());
        //cv::rectangle(img2, rect, cv::Scalar(255, 0, 0), 1);
    }
    std::vector<cv::Mat> bmtxts;
    for (auto& mtxt:mtxts)
    {
        cv::Mat grayImage;
        // 将彩色图片转换为灰度图片
        cv::cvtColor(mtxt, grayImage, cv::COLOR_BGR2GRAY);
        cv::Mat binaryImage = cv::Mat::zeros(grayImage.size(), grayImage.type());
        cv::threshold(grayImage, binaryImage, 250, 255, cv::THRESH_BINARY);
        cv::Mat threeChannelImage;
        std::vector<cv::Mat> channels = { binaryImage,  binaryImage,  binaryImage };
        cv::merge(channels, threeChannelImage);
        bmtxts.push_back(threeChannelImage);
    }
    for (size_t i = 0; i < bmtxts.size(); i++)
    {
        cv::imshow(std::to_string(i), bmtxts[i]);
    }
    // 显示结果
    cv::imshow("Contours", img2);
    cv::waitKey(0);
}

int test6(char** argv) {
    //-- 读取图像
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    //-- 初始化
    // 定义了两个std::vector<KeyPoint>类型的变量keypoints_1和keypoints_2，用于保存检测到的特征点。
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    // 定义了两个cv::Mat类型的变量descriptors_1和descriptors_2，用于保存提取到的特征描述子。
    cv::Mat descriptors_1, descriptors_2;

    // 创建了一个cv::Ptr<FeatureDetector>类型的指针detector，使用cv::ORB::create()函数创建了ORB特征检测器。
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();

    // 创建了一个cv::Ptr<DescriptorExtractor>类型的指针descriptor，同样使用cv::ORB::create()函数创建了ORB特征描述子提取器。
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    // 创建了一个cv::Ptr<DescriptorMatcher>类型的指针matcher，
    // 使用cv::DescriptorMatcher::create("BruteForce-Hamming")函数创建了一个使用汉明距离的暴力匹配器。
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");



    //-- 第一步:检测 Oriented FAST 角点位置
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1); // 对 img_1 进行特征点检测，并将检测到的特征点保存在 keypoints_1 中。
    detector->detect(img_2, keypoints_2); // 对 img_2 进行特征点检测，并将检测到的特征点保存在 keypoints_2 中。

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    // 对 img_1 中的特征点 keypoints_1 进行特征描述子提取，并将提取到的特征描述子保存在 descriptors_1 中。
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    // 对 img_2 中的特征点 keypoints_2 进行特征描述子提取，并将提取到的特征描述子保存在 descriptors_2 中。
    descriptor->compute(img_2, keypoints_2, descriptors_2);


    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds. " << std::endl;


    /**
     * 通过调用 cv::drawKeypoints 函数来绘制特征点。函数的参数依次为：
     * img_1：输入的原始图像。
     * keypoints_1：特征点的位置信息
     * outimg1：输出的绘制特征点后的图像
     * cv::Scalar::all(-1)：绘制特征点的颜色，这里使用 -1 表示随机颜色。
     * cv::DrawMatchesFlags::DEFAULT：绘制特征点的附加标志，这里使用默认值。
    */

    cv::Mat outimg1; // 用于存储绘制特征点后的图像
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB features", outimg1);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    // matches 用于匹配关键点描述符的类查询描述符索引、训练描述符索引、训练图像索引和描述符之间的距离。
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    // 使用 matcher 对 descriptors_1 和 descriptors_2 进行特征描述子匹配，并将匹配结果保存在matches中。
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds. " << std::endl;

    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    /**
     * 首先，通过调用 minmax_element 函数，传入 matches.begin() 和 matches.end() 迭代器范围，
     * 使用 lambda 表达式 [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; } 作为比较函数，
     * 来找到 matches 中距离最小和最大的元素。
     * 然后，使用 min_max.first 获取最小值的迭代器，并通过 ->distance 获取对应的距离值，将其赋值给 min_dist 变量。
     * 接下来，使用 min_max.second 获取最大值的迭代器，并通过 ->distance 获取对应的距离值，将其赋值给 max_dist 变量。
    */

    auto min_max = std::minmax_element(matches.begin(), matches.end(),
        [](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 40.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    //-- 第五步:绘制匹配结果
    cv::Mat img_match; // 创建一个空的图像用于存储所有匹配的关键点
    cv::Mat img_goodmatch;  // 创建一个空的图像用于存储好的匹配的关键点

    // 绘制所有匹配的关键点
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);

    // 绘制好的匹配的关键点
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);

    std::cout << (float)good_matches.size() / (float)matches.size();

    cv::imshow("all matches", img_match);
    cv::imshow("good matches", img_goodmatch);
    cv::waitKey(0);

    return 0;
}

void test7() {
    // 加载原始图片
    cv::Mat originalImage = cv::imread("1.PNG");

    // 检查图片是否成功加载
    if (originalImage.empty()) {
        std::cerr << "Error: Loading image" << std::endl;
        return ;
    }

    // 创建一个Mat对象用于存储改变大小后的图片
    cv::Mat resizedImage;

    // 设定新的图片大小
    cv::Size newSize(21, 21); // 例如改变为640x480大小

    // 改变图片大小
    cv::resize(originalImage, resizedImage, newSize);

    cv::Mat img = cv::imread("20240901015503.png");
    cv::Mat subimg = img(cv::Rect(0, 507, 190, 60));
    PictureMatch pm;
    auto r= pm.getmatchbox(subimg, resizedImage);
    std::cout << r.pixsc << std:: endl;
    cv::rectangle(subimg, r.getp1(), r.getp2(), cv::Scalar(0, 255, 0));//画出轮廓范围
    // 保存改变大小后的图片
    cv::imshow("test_image", subimg);
    cv::imshow("resized_image", resizedImage);
    cv::waitKey(0);
}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
