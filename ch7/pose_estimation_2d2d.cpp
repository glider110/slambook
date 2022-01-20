#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <list>
#include <chrono>
// #include "extra.h" // use this if in OpenCV2 
using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void pose_estimation_2d2d (
    std::vector<KeyPoint> keypoints_1,
    std::vector<KeyPoint> keypoints_2,
    std::vector< DMatch > matches,
    Mat& R, Mat& t );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    //光流计算关键点
    #if 1

        int maxCount = 50;//最大特征点数
        double minDis = 20;//：对于初选出的角点而言，如果
        //在其周围minDis范围内存在其他更强角点，则将此角点删除
        double qLevel = 0.01;//角点的品
 
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; 
        vector<cv::Point2f> prev_keypoints;
        vector<unsigned char> status;
        vector<float> error; 
        Mat img_1_gray;
        cout<<"prev_keypoints: "<<prev_keypoints.size()<<endl;
        cvtColor(img_1, img_1_gray, CV_RGB2GRAY);
        goodFeaturesToTrack(img_1_gray, prev_keypoints, maxCount, qLevel, minDis);//角点识别
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        cout<<"prev_keypoints: "<<prev_keypoints.size()<<endl;
        cout<<"next_keypoints: "<<next_keypoints.size()<<endl;
        cout<<"status: "<<status.size()<<endl;
            //-- 计算基础矩阵
        Mat fundamental_matrix;
        fundamental_matrix = findFundamentalMat ( prev_keypoints, next_keypoints, CV_FM_8POINT );
        cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

        //-- 计算本质矩阵
        Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
        double focal_length = 521;			//相机焦距, TUM dataset标定值
        Mat essential_matrix;
        essential_matrix = findEssentialMat ( prev_keypoints, next_keypoints, focal_length, principal_point );
        cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

        //-- 计算单应矩阵
        Mat homography_matrix;
        homography_matrix = findHomography ( prev_keypoints, next_keypoints, RANSAC, 3 );
        cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

        //-- 从本质矩阵中恢复旋转和平移信息.
        Mat R_optical,t_optical;
        recoverPose ( essential_matrix, prev_keypoints, next_keypoints, R_optical, t_optical, focal_length, principal_point );
        cout<<"R is "<<endl<<R_optical<<endl;
        cout<<"t is "<<endl<<t_optical<<endl;
        Mat img_2_copy2 = img_2.clone();
        for (int i = 0; i < next_keypoints.size(); i++)
		{
			circle(img_2_copy2, next_keypoints[i], 3, Scalar(0, 255, 0), 2);
			line(img_2_copy2, prev_keypoints[i], next_keypoints[i], Scalar(255, 0, 0), 2);//画出两帧之间的光流变化
		}
		
        namedWindow("OPTICAL",CV_WINDOW_AUTOSIZE);
        imshow("OPTICAL", img_2_copy2);
        waitKey(100000);
     #endif


    

    // -- 估计两张图像间运动
    Mat R,t;
    Mat R_vec;
    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );
    Rodrigues(R, R_vec);
    cout << "Rodrigues = " << R_vec << endl;
    cout << "roll degree = " << R_vec*180/ CV_PI << endl;


    //-- 验证E=t^R*scale
    Mat t_x = ( Mat_<double> ( 3,3 ) <<
                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
                -t.at<double> ( 1,0 ),     t.at<double> ( 0,0 ),      0 );

    cout<<"t^R="<<endl<<t_x*R<<endl;

    //-- 验证对极约束
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    for ( DMatch m: matches )
    {
        Point2d pt1 = pixel2cam ( keypoints_1[ m.queryIdx ].pt, K );
        Mat y1 = ( Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        Point2d pt2 = pixel2cam ( keypoints_2[ m.trainIdx ].pt, K );
        Mat y2 = ( Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }


    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }
    cout<<"一共找到了XXX"<<points1.size() <<"组匹配点"<<endl;
    cout<<"一共找到了XXX"<<points2.size() <<"组匹配点"<<endl;
    Mat img_2_copy1 = img_2.clone();
    for (int i = 0; i < points2.size(); i++)
    {
        circle(img_2_copy1, points2[i], 3, Scalar(0, 255, 0), 2);
        line(img_2_copy1, points1[i], points2[i], Scalar(0, 0, 255), 2);//画出两帧之间的光流变化
    }
    
    namedWindow("ORB",CV_WINDOW_AUTOSIZE);
    imshow("ORB", img_2_copy1);
    // waitKey(0);
}


Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}


void pose_estimation_2d2d ( std::vector<KeyPoint> keypoints_1,
                            std::vector<KeyPoint> keypoints_2,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( 325.1, 249.7 );	//相机光心, TUM dataset标定值
    double focal_length = 300;			//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
    
}
