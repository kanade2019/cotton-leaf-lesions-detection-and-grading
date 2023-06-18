#include<opencv2/opencv.hpp>
#include<iostream>
#include"fit.h"
#include<vector>
#include<fstream>
#include<time.h>

using namespace std;
using namespace cv;

Mat window(Mat InputArray, Mat InputTemplate, unsigned short mode=1);
Mat cut(Mat res, Mat InputArray, Size sp);

int main()
{
	//namedWindow("1", WINDOW_AUTOSIZE);
	ofstream data("time2.txt", ios::trunc);
	string filepath = "D:/study/lib/棉花/小论文/data/cut_image/";
	string outpath = "D:/study/lib/棉花/小论文/data/t/";
	//string savepath = "D:/study/lib/棉花/小论文/";
	vector<string> fileindex;
	//glob(filepath, fileindex);
	glob("D:/study/lib/棉花/小论文/data/restart/", fileindex);
	vector<string> filenum;
	for (vector<string>::iterator first = fileindex.begin(); first != fileindex.end(); first++)
		if (first->find("RGB.JPG") != string::npos)
			filenum.push_back(first->substr(first->rfind("\\") + 1, first->rfind("RGB.JPG") - first->rfind("\\") - 1));
	for (vector<string>::iterator first = filenum.begin(); first != filenum.end(); first++)
	{
		Mat rgb = imread("D:/study/lib/棉花/小论文/data/restart/" + *first + "RGB.JPG");
		Mat gre = imread(filepath + *first + "GRE.TIF", 0);
		Mat red = imread(filepath + *first + "RED.TIF", 0);
		Mat reg = imread(filepath + *first + "REG.TIF", 0);
		Mat nir = imread(filepath + *first + "NIR.TIF", 0);
		/*resize(rgb, rgb, Size(0, 0), 1/3.6, 1/3.6);
		resize(gre, gre, Size(0, 0), 1 / 3.6, 1 / 3.6);
		resize(red, red, Size(0, 0), 1 / 3.6, 1 / 3.6);
		resize(reg, reg, Size(0, 0), 1 / 3.6, 1 / 3.6);
		resize(nir, nir, Size(0, 0), 1 / 3.6, 1 / 3.6);*/
		Mat rgbs[3];
		split(rgb, rgbs);
		cvtColor(rgb, rgb, COLOR_BGR2GRAY);
		Size sp = rgb.size();
		rgbs[0].convertTo(rgbs[0], CV_64F);
		rgbs[1].convertTo(rgbs[1], CV_64F);
		rgbs[2].convertTo(rgbs[2], CV_64F);
		rgb.convertTo(rgb, CV_64F);
		Mat Togre = Mat::zeros(sp, CV_64F);
		Togre = rgbs[1] - rgbs[0].mul(0.5);
		normalize(Togre, Togre, 0, 255, NORM_MINMAX);
		Togre.convertTo(Togre, CV_8U);
		/*Mat Tored = Mat::zeros(sp, CV_64F);
		Tored = rgbs[2] - 0.5 * rgbs[0];
		normalize(Tored, Tored, 0, 255, NORM_MINMAX);
		Tored.convertTo(Tored, CV_8U);*/
		Mat Tonir_reg = Mat::zeros(sp, CV_64F);
		Tonir_reg = rgb;
		normalize(Tonir_reg, Tonir_reg, 0, 255, NORM_MINMAX);
		Tonir_reg.convertTo(Tonir_reg, CV_8U);

		rgbs[2].convertTo(rgbs[2], CV_8U);

		clock_t  start = clock();
		try{
			cout << *first << endl;
			// data << *first;
			gre = window(gre, Togre);
			cout << endl << double(clock() - start) / CLOCKS_PER_SEC << endl;
			red = window(red, rgbs[2], 0);
			cout << endl << double(clock() - start) / CLOCKS_PER_SEC << endl;
			reg = window(reg, Tonir_reg, 7);
			cout << endl << double(clock() - start) / CLOCKS_PER_SEC << endl;
			nir = window(nir, Tonir_reg, 0);
			cout << endl << double(clock() - start) / CLOCKS_PER_SEC << endl;

			/*resize(gre, gre, Size(0, 0), 3.6, 3.6);
			resize(red, red, Size(0, 0), 3.6, 3.6);
			resize(reg, reg, Size(0, 0), 3.6, 3.6);
			resize(nir, nir, Size(0, 0), 3.6, 3.6);*/
			imwrite(outpath + *first + "GRE.TIF", gre);
			imwrite(outpath + *first + "RED.TIF", red);
			imwrite(outpath + *first + "REG.TIF", reg);
			imwrite(outpath + *first + "NIR.TIF", nir);
		}
		catch (cv::Exception e)
		{
			cout << e.what() << endl;
		}
		//getchar();
	}
	data.close();
	return 0;
}

Mat window(Mat InputArray,Mat InputTemplate, unsigned short mode)
{
	Size sp = InputTemplate.size();
	Mat res = Mat::zeros(Size(InputArray.cols - sp.width + 1, InputArray.rows - sp.height + 1), CV_32F);
	//Mat res = Mat_<double>(InputArray.cols - sp.width + 1, InputArray.rows - sp.height + 1);
	//matchTemplate(InputArray, InputTemplate, res, TM_CCOEFF_NORMED);
	//string savepath = "D:/study/lib/棉花/小论文/data/t/1/";
	//int num = 1;
	stringstream num_s;
	Mat dst = Mat::zeros(sp, CV_8U);
	for (int i = 0; i + sp.width < InputArray.cols; i++)
	{
		double process = i / double((InputArray.cols - sp.width)) * 100;
		// cout << "\r\b" << fixed <<  setw(5) << process << "%";
		for (int j = 0; j + sp.height < InputArray.rows; j++)
		{
			dst = InputArray(Rect(i, j, sp.width, sp.height));
			if (dst.empty())
				continue;
			fit dst_InputArray(InputTemplate, dst, mode);
			dst_InputArray.line_fit();

			//num_s.str("");
			//num_s << num;
			//imwrite(savepath + num_s.str() + ".jpg", dst_InputArray.plot(false));
			//num++;

			res.at<float>(j, i) += dst_InputArray.r2;
		}
	}
	//Mat temp = Mat::zeros(Size(InputArray.cols - sp.width + 1, InputArray.rows - sp.height + 1), CV_8UC1);
	//temp = res*255;
	//imwrite("D:/study/lib/棉花/小论文/data/t/1/0.bmp", temp);
	//getchar();
	//imshow("res", res);
	//waitKey(100);
	return cut(res, InputArray, sp);
}

Mat cut(Mat res, Mat InputArray, Size sp)
{
	double maxVal, minVal;
	Point minLoc, maxLoc;
	minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);
	Rect rect = Rect(maxLoc.x, maxLoc.y, sp.width, sp.height);
	cout << rect;
	return InputArray(rect);
}