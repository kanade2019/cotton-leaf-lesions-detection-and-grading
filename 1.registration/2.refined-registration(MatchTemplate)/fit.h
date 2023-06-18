#pragma once

#ifndef __OPENCV_HPP__
#define __OPENCV_HPP__
#include<opencv2/opencv.hpp>
#endif

#ifndef __IOSTREAM_H__
#define __IOSTREAM_H__
#include<iostream>
#endif

#ifndef __VECTOR_H__
#define __VECTOR_H__
#include<vector>
#endif

#ifndef __UNORDERED_MAP_H__
#define __UNORDERED_MAP_H__
#include <unordered_map>
#endif

using namespace std;
using namespace cv;


bool cmp(Point const& a, Point const& b)//排序条件
{
	return a.x < b.x;
}

class fit
{
public:
	fit(Mat img1, Mat img2, unsigned short mode);
	~fit() { waitKey(0); }
private:
	void remove_dumplication(vector<Point>& p);//去重
	Mat polyfit(vector<cv::Point>& in_point, int n);//多项式拟合
	double R2(Mat k, vector<Point>& point);//回归系数
public:
	vector<Point> point;//去重、排序后的点集
	Mat multinomial;//多项式
	double r2;//相关系数
	int best_fit();//寻找拟合最优解
	double line_fit();
	Mat plot(bool flage = false);//绘图
	unsigned short polyfit_mode=1;//多项式拟合的阶数
};

fit::fit(Mat img1, Mat img2, unsigned short mode)
{
	polyfit_mode = mode;
	//vector<Point> original_point;
	unordered_map<unsigned char, unsigned char> original_point;
	for (MatConstIterator_<uchar> first1=img1.begin<uchar>(),first2=img2.begin<uchar>();first1!=img1.end<uchar>()&&first2!=img2.end<uchar>();first1++,first2++)
	{
		if (*first1 == 0 || *first2 == 0)
			continue;
		//original_point.push_back(Point(*first1, *first2));
		original_point[*first1] = *first2;
	}
	//sort(original_point.begin(), original_point.end(), cmp);
	//remove_dumplication(original_point);
	for (auto& p : original_point) {
		point.push_back(Point(p.first, p.second));
	}
}

Mat fit::polyfit(vector<cv::Point>& in_point, int n)//多项式拟合
{
	int size = in_point.size();
	int x_num = n + 1;
	Mat mat_u(size, x_num, CV_64F);
	Mat mat_y(size, 1, CV_64F);

	/*for (int i = 0; i < mat_u.rows; ++i)
		for (int j = 0; j < mat_u.cols; ++j)
		{
			mat_u.at<double>(i, j) = pow(in_point[i].x, j);
		}*/
	int j = 0;
	for (MatIterator_<double> first = mat_u.begin<double>(); first != mat_u.end<double>(); first++,j++)
		*first = pow(in_point[j / mat_u.cols].x, j % mat_u.cols);
	/*for (int i = 0; i < mat_y.rows; ++i)
	{
		mat_y.at<double>(i, 0) = in_point[i].y;
	}*/
	auto first2 = in_point.begin();
	for (MatIterator_<double> first = mat_y.begin<double>(); first != mat_y.end<double>()&&first2 != in_point.end(); first++, first2++)
		*first = first2->y;

	Mat mat_k(x_num, 1, CV_64F);
	mat_k = (mat_u.t() * mat_u).inv() * mat_u.t() * mat_y;

	return mat_k;
}

int fit::best_fit()//寻找拟合最优解
{
	int n = 1;
	double t = 0;
	Mat kt;
	while (true)
	{
		kt = polyfit(point, n);
		t = R2(kt, point);
		if (r2 > t)
			break;
		multinomial = kt;
		r2 = t;
		n++;
	}
	return n;
}

double fit::line_fit()
{
	if (polyfit_mode == 0)
		best_fit();
	else {
		multinomial = polyfit(point, polyfit_mode);
		r2 = R2(multinomial, point);
	}
	//best_fit();
	return r2;
}

double fit::R2(Mat k, vector<Point>& point)//回归系数
{
	double mean = 0;
	for (vector<Point>::iterator first = point.begin(); first != point.end(); first++)
		mean += first->y;
	mean /= point.size();
	double SSres = 0;
	double SStot = 0;
	for (vector<Point>::iterator first = point.begin(); first != point.end(); first++)
	{
		SStot += pow(first->y - mean, 2);
		double y = 0;
		for (int i = 0; i < k.rows; i++)
			y += k.at<double>(i, 0) * pow(first->x, i);
		SSres += pow(first->y - y, 2);
	}
	double r = 1 - SSres / SStot;
	//cout << r << endl;
	return r;
}

void fit::remove_dumplication(vector<Point>& p)//去重
{
	vector<Point> p2;
	for (auto first = p.begin(); first != p.end(); first++)
	{
		if (first->x == (first + 1)->x)
			continue;
		else
			p2.push_back(*first);
	}
	p = p2;
}

Mat fit::plot(bool flage)
{
	Mat map = Mat::zeros(Size(255, 255), CV_8UC3);
	map = Scalar(255, 255, 255) - map;
	for (int i = 0; i < point.size(); i++)
	{
		circle(map, point[i], 1, Scalar(0, 0, 255), -1);
	}
	for (int x = 0; x <= 255; x++)
	{
		double y = 0;
		for (int i = 0; i < multinomial.rows; i++)
			y += multinomial.at<double>(i, 0) * pow(x, i);
		circle(map, Point(x, y), 1, Scalar(255, 0, 0), -1);
	}
	if (flage)
		imshow("1", map);
	return map;
}
