
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>


using namespace cv;
using namespace std;


class shapeInfoProducer
{

public:

	shapeInfoProducer(cv::Mat &src, float magnitude, float threshold);
	cv::Mat srcImg;
	cv::Mat magnitudeImg;		//�ݶȷ�ֵͼ
	cv::Mat quantized_angle;				// ������ĽǶ�ͼ�� [0-7]��
	cv::Mat angle_ori;			// �Ƕ�ͼ
	float magnitude_value;			//ѡ������ʱ�ķ�ֵ��ֵ
	float score_threshold;		//������Ŷ���ֵ

	//���������ݶ�->ת��������->�㲥->ѡ������
	void quantizedOrientations();



};


void hysteresisGradient(Mat &magnitude, Mat &quantized_angle, Mat &angle, float threshold)
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	Mat_<unsigned char> quantized_unfiltered;
	angle.convertTo(quantized_unfiltered, CV_8U, 16 / 360.0);

	// Zero out top and bottom rows
	/// @todo is this necessary, or even correct?
	memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
	memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
	// Zero out first and last columns
	for (int r = 0; r < quantized_unfiltered.rows; ++r)
	{
		quantized_unfiltered(r, 0) = 0;
		quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
	}

	// Mask 16 buckets into 8 quantized orientations
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 1; c < angle.cols - 1; ++c)
		{
			quant_r[c] &= 7;// �����������һ����ת���緽��15��תΪ7
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	quantized_angle = Mat::zeros(angle.size(), CV_8U);
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		float *mag_r = magnitude.ptr<float>(r);

		for (int c = 1; c < angle.cols - 1; ++c)
		{
			if (mag_r[c] > threshold)
			{
				// Compute histogram of quantized bins in 3x3 patch around pixel
				int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

				uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1); // ̫������
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;

				// Find bin with the most votes from the patch
				int max_votes = 0;
				int index = -1;
				for (int i = 0; i < 8; ++i)
				{
					if (max_votes < histogram[i])
					{
						index = i;
						max_votes = histogram[i];
					}
				}

				// Only accept the quantization if majority of pixels in the patch agree
				static const int NEIGHBOR_THRESHOLD = 5;
				if (max_votes >= NEIGHBOR_THRESHOLD)
					quantized_angle.at<uchar>(r, c) = uchar(1 << index);
			}
		}
	}
}


shapeInfoProducer::shapeInfoProducer(cv::Mat &in_src, float in_magnitude, float in_threshold)
{
	this->srcImg = in_src;
	this->magnitude_value = in_magnitude;
	this->score_threshold = in_threshold;
	cout << "srcImg.rows:" << this->srcImg.rows << endl;

}

void shapeInfoProducer::quantizedOrientations()
{

	Mat smoothed;
	static const int KERNEL_SIZE = 7;
	GaussianBlur(this->srcImg, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

	cv::Mat sobel_dx, sobel_dy;
	Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
	Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
	magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
	phase(sobel_dx, sobel_dy, angle_ori, true);
	// ������8������
	hysteresisGradient(magnitudeImg, quantized_angle, angle_ori, magnitude_value * magnitude_value);
	//Mat temp_show_angle = this->quantized_angle.clone();
	//Mat temp_show_angle_ori = this->angle_ori.clone();


}



void test_shape_match()
{
	cv::Mat img = cv::imread("F:\\1heils\\shape_based_matching\\test\\case1\\train_new.png", 0);//�涨������ǻҶ�ͼ����ͨ�����Ȳ�Ū
	shapeInfoProducer trainer(img, 100, 100);
	trainer.quantizedOrientations();
}


int main()
{
	test_shape_match();

	return 0;
}