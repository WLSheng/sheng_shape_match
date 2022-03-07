
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <omp.h>
#include <set>

using namespace cv;
using namespace std;

struct Feature
{
	int x;
	int y;
	int label;
	float theta;

	void read(const cv::FileNode& fn);
	//void write(cv::FileStorage &fs) const;

	Feature() : x(0), y(0), label(0) {}
	Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

void Feature::read(const FileNode& fn)
{
	FileNodeIterator fni = fn.begin();
	fni >> x >> y >> label;
}

struct Candidate
{
	Candidate(int x, int y, int label, float score);

	/// Sort candidates with high score to the front
	bool operator<(const Candidate& rhs) const
	{
		return score > rhs.score;
	}

	Feature f;
	float score;
};
inline Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}


class shapeInfoProducer
{

public:
	shapeInfoProducer(cv::Mat& src, int in_featuresNum, float magnitude, string inpath);
	//shapeInfoProducer(cv::Mat& src, int in_featuresNum, float magnitude, float threshold) {};
	cv::Mat srcImg;
	cv::Mat magnitudeImg;		//梯度幅值图
	cv::Mat quantized_angle;				// 量化后的角度图，根据离散量化投票机制，找出每个位置3x3领域中出现次数最多的方向(为了避免小的形变的影响)，如果投票数量小于5，方向则为0， [0-7]；
	cv::Mat angle_ori;			// 角度图, 
	float magnitude_value;			//选特征点时的幅值阈值
	int num_features;
	std::string path;

	std::vector<Feature> out_features;	// 过了极大值，然后再过一次距离判断，找到的最终特征点
	static inline int getLabel(int quantized)
	{
		switch (quantized)
		{
		case 1:
			return 0;
		case 2:
			return 1;
		case 4:
			return 2;
		case 8:
			return 3;
		case 16:
			return 4;
		case 32:
			return 5;
		case 64:
			return 6;
		case 128:
			return 7;
		default:
			CV_Error(Error::StsBadArg, "Invalid value of quantized parameter");
			return -1; //avoid warning
		}
	}

	//训练步骤：梯度->转方向量化->广播->选特征点
	void train();



	// 滞后方向，就是找出3X3领域内主要的方向，
	// magnitudeImg, quantized_angle, angle_ori, magnitude_value * magnitude_value
	void hysteresisGradient();

	bool extractFeaturePoints();

	bool selectScatteredFeatures(const std::vector<Candidate>& candidates, std::vector<Feature>& features,
		size_t num_features, float distance);


};


void shapeInfoProducer::hysteresisGradient()//Mat &magnitude, Mat &quantized_angle, Mat &angle, float threshold
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	Mat_<unsigned char> quantized_unfiltered;
	this->angle_ori.convertTo(quantized_unfiltered, CV_8U, 16 / 360.0);

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
	for (int r = 1; r < this->angle_ori.rows - 1; ++r)
	{
		uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 1; c < this->angle_ori.cols - 1; ++c)
		{
			quant_r[c] &= 7;// 很巧妙地做了一个反转，如方向15的转为7,这里就是量化方向，360度换乘16块，再变成8块，也就是8个方向
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	quantized_angle = Mat::zeros(this->angle_ori.size(), CV_8U);
	float strong_magnitude_value = this->magnitude_value * this->magnitude_value;
	for (int r = 1; r < this->angle_ori.rows - 1; ++r)
	{
		float* mag_r = this->magnitudeImg.ptr<float>(r);

		for (int c = 1; c < this->angle_ori.cols - 1; ++c)
		{
			if (mag_r[c] > strong_magnitude_value)
			{
				// Compute histogram of quantized bins in 3x3 patch around pixel
				int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

				uchar* patch3x3_row = &quantized_unfiltered(r - 1, c - 1); // 太巧妙了
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
					this->quantized_angle.at<uchar>(r, c) = uchar(1 << index);
			}
		}
	}
}


bool shapeInfoProducer::extractFeaturePoints()
{

	std::vector<Candidate> candidates;
	float threshold_sq = this->magnitude_value * this->magnitude_value;

	int nms_kernel_size = 9 / 2;
	cv::Mat magnitude_valid = cv::Mat(this->magnitudeImg.size(), CV_8UC1, cv::Scalar(255));
	cv::Mat temp_show = this->quantized_angle.clone();

	// 非极大值抑制, 找到比上下左右都大且大于某个阈值的像素。
	cv::Mat left = cv::Mat::zeros(this->magnitudeImg.size(), this->magnitudeImg.type());
	cv::Mat right = cv::Mat::zeros(this->magnitudeImg.size(), this->magnitudeImg.type());
	cv::Mat top = cv::Mat::zeros(this->magnitudeImg.size(), this->magnitudeImg.type());
	cv::Mat bottom = cv::Mat::zeros(this->magnitudeImg.size(), this->magnitudeImg.type());
	int nms_offset = 1;
	int rows = this->magnitudeImg.rows;
	int cols = this->magnitudeImg.cols;
	this->magnitudeImg.rowRange(0, rows - nms_offset).copyTo(top.rowRange(nms_offset, rows));
	this->magnitudeImg.rowRange(nms_offset, rows).copyTo(bottom.rowRange(0, rows - nms_offset));

	this->magnitudeImg.colRange(0, cols - nms_offset).copyTo(left.colRange(nms_offset, cols));
	this->magnitudeImg.colRange(nms_offset, cols).copyTo(right.colRange(0, cols - nms_offset));


	cv::Mat binary = this->magnitudeImg >= threshold_sq
		& this->magnitudeImg > left
		& this->magnitudeImg > right
		& this->magnitudeImg > top
		& this->magnitudeImg > bottom;



	std::vector<Candidate> temp_candidates;
	for (int row = 1; row < binary.rows - 1; ++row) {
		for (int col = 1; col < binary.cols - 1; ++col)
		{
			if (binary.at<uint8_t>(row, col) != 0 && (int)this->quantized_angle.at<uchar>(row, col) != 0)
			{
				float _score = this->magnitudeImg.at<float>(row, col);
				//cout << "col:" << col << ", row: " << row << ", angle:" << (int)this->quantized_angle.at<uchar>(row, col) << endl;
				temp_candidates.emplace_back(Candidate(col, row, this->getLabel(this->quantized_angle.at<uchar>(row, col)), _score));
				//index_score.emplace_back(_temp);

			}
		}
	}
	sort(temp_candidates.begin(), temp_candidates.end()); // 降序

	std::vector<int> del_index;
	for (int i = 0; i < temp_candidates.size() - 1; i++)
	{
		for (int ii = i + 1; ii < temp_candidates.size(); ii++)
		{
			cv::Point one_coor = cv::Point(temp_candidates.at(i).f.x, temp_candidates.at(i).f.y);
			cv::Point two_coor = cv::Point(temp_candidates.at(ii).f.x, temp_candidates.at(ii).f.y);
			if (abs(one_coor.x - two_coor.x) <= nms_kernel_size && abs(one_coor.y - two_coor.y) <= nms_kernel_size)
			{
				// 遇到领域内重复,并且前面是必定大于后面的
				del_index.emplace_back(ii);
			}
		}
	}
	sort(del_index.begin(), del_index.end()); // 降序
	del_index.erase(unique(del_index.begin(), del_index.end()), del_index.end());

	for (int i = 0; i < del_index.size(); i++)
	{
		temp_candidates.erase(temp_candidates.begin() + (del_index.at(i) - i));

	}

	// 临时可视化全部极大值特征
	cv::Mat show_max;
	cvtColor(this->srcImg, show_max, COLOR_GRAY2RGB);
	for (int i = 0; i < temp_candidates.size(); i++)
	{
		cv::Point show_coor = cv::Point(temp_candidates.at(i).f.x, temp_candidates.at(i).f.y);
		//show_max.at<Vec3b>(show_coor.y, show_coor.x) = cv::Scalar(0, 255, 0);
		show_max.at<Vec3b>(show_coor.y, show_coor.x)[0] = 0;
		show_max.at<Vec3b>(show_coor.y, show_coor.x)[1] = 255;
		show_max.at<Vec3b>(show_coor.y, show_coor.x)[2] = 255;
	}


	// 原始的代码有问题，用自己思路的找极大值方法
	/*
	for (int r = 0 + nms_kernel_size / 2; r < this->magnitudeImg.rows - nms_kernel_size / 2; ++r)
	{

		for (int c = 0 + nms_kernel_size / 2; c < this->magnitudeImg.cols - nms_kernel_size / 2; ++c)
		{
			float score = 0;
			if (magnitude_valid.at<uchar>(r, c) > 0) {
				score = this->magnitudeImg.at<float>(r, c);
				bool is_max = true;
				for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++) {
					for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) {
						if (r_offset == 0 && c_offset == 0) continue;

						if (score < this->magnitudeImg.at<float>(r + r_offset, c + c_offset)) {
							score = 0;
							is_max = false;
							break;
						}
					}
					if (!is_max) break;
				}

				if (is_max) {
					for (int r_offset = -nms_kernel_size / 2; r_offset <= nms_kernel_size / 2; r_offset++) {
						for (int c_offset = -nms_kernel_size / 2; c_offset <= nms_kernel_size / 2; c_offset++) {
							if (r_offset == 0 && c_offset == 0) continue;
							magnitude_valid.at<uchar>(r + r_offset, c + c_offset) = 0;
						}
					}
				}
			}

			if (score > threshold_sq && this->angle_ori.at<uchar>(r, c) > 0)
			{
				candidates.push_back(Candidate(c, r, this->getLabel(this->angle_ori.at<uchar>(r, c)), score));
				candidates.back().f.theta = angle_ori.at<float>(r, c);
			}
		}
	}
	*/

	// We require a certain number of features
	if (temp_candidates.size() < num_features)
	{
		if (temp_candidates.size() <= 4) {
			std::cout << "too few features, abort" << std::endl;
			return false;
		}
		std::cout << "have no enough features, exaustive mode" << std::endl;
	}

	// NOTE: Stable sort to agree with old code, which used std::list::sort()
	std::stable_sort(temp_candidates.begin(), temp_candidates.end());

	// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
	// 初始距离阈值采用基于窄轮廓候选剩余量的启发式算法;
	// 一步步地把特征压缩到输入指定的输入特征数量，准则：每个特征相邻距离足够远
	float distance = static_cast<float>(temp_candidates.size() / this->num_features + 1);

	if (!selectScatteredFeatures(temp_candidates, this->out_features, num_features, distance))
	{
		return false;
	}

	// 临时可视化极大值特征
	cv::Mat show_two;
	cvtColor(this->srcImg, show_two, COLOR_GRAY2RGB);
	for (int i = 0; i < this->out_features.size(); i++)
	{
		cv::Point show_coor = cv::Point(this->out_features.at(i).x, this->out_features.at(i).y);
		//show_max.at<Vec3b>(show_coor.y, show_coor.x) = cv::Scalar(0, 255, 0);
		show_two.at<Vec3b>(show_coor.y, show_coor.x)[0] = 0;
		show_two.at<Vec3b>(show_coor.y, show_coor.x)[1] = 255;
		show_two.at<Vec3b>(show_coor.y, show_coor.x)[2] = 255;
	}


	return true;
}


bool shapeInfoProducer::selectScatteredFeatures(const std::vector<Candidate>& candidates, std::vector<Feature>& features,
	size_t num_features, float distance)
{
	// 目的：均衡（分散）提取特征
	// 思路：第一个特征是肯定保留得，放进features中，然后从candidates提取下一个，和features的全部进行比较，只有下一个与features中的全部进行距离比较，都大于distance_sq才放进features，循环比较；
	// 退出逻辑：在candidates中判断到最后一个，看看features的数量是否逼近指定的特征数量，如果还是大于特征数量，对距离加一，然后再从0开始判断；
	// 直到拿到的features数量少于指定的特征数量，才退回上一次的距离，拿上一次的features

	features.clear();
	float distance_sq = distance * distance;
	int i = 0;

	bool first_select = true;

	while (true)
	{
		Candidate c = candidates[i];

		// Add if sufficient distance away from any previously chosen feature
		bool keep = true;
		for (int j = 0; (j < (int)features.size()) && keep; ++j)
		{
			Feature f = features[j];
			keep = (c.f.x - f.x) * (c.f.x - f.x) + (c.f.y - f.y) * (c.f.y - f.y) >= distance_sq;
		}
		if (keep)
			features.push_back(c.f);

		if (++i == (int)candidates.size()) {
			bool num_ok = features.size() >= num_features;

			if (first_select) {
				if (num_ok) {
					features.clear(); // we don't want too many first time
					i = 0;
					distance += 1.0f;
					distance_sq = distance * distance;
					continue;
				}
				else {
					first_select = false;// 往回退一次
				}
			}

			// Start back at beginning, and relax required distance
			i = 0;
			distance -= 1.0f;
			distance_sq = distance * distance;
			if (num_ok || distance < 3) {
				break;
			}
		}
	}
	return true;
}


shapeInfoProducer::shapeInfoProducer(cv::Mat& in_src, int in_featuresNum, float in_magnitude, string inpath)
{
	this->srcImg = in_src;
	this->magnitude_value = in_magnitude;
	this->num_features = in_featuresNum;
	this->path = inpath;
	cout << "strat train, train img rows:" << this->srcImg.rows << endl;
	cout << "num_features:" << num_features << endl;
	cout << "magnitude_value:" << magnitude_value << endl;
	cout << "save feature path:" << this->path << endl;

}


void shapeInfoProducer::train()
{

	Mat smoothed;
	static const int KERNEL_SIZE = 3;
	cv::GaussianBlur(this->srcImg, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

	cv::Mat sobel_dx, sobel_dy;
	cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
	cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
	magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
	cv::phase(sobel_dx, sobel_dy, angle_ori, true);
	// 量化到8个方向，再根据领域找出现次数最多的方向，结果放在：quantized_angle
	hysteresisGradient();

	// todo 这里应加个循环的提取，金字塔和旋转的模板
	//  nms 根据幅值图筛选极大值特征点

	if (!this->extractFeaturePoints())
	{
		//return -1;
		cout << "提取失败" << endl;

	}
	else
	{
		// 保存训练的特征
		cv::FileStorage fs(this->path, cv::FileStorage::WRITE);
		fs << "pyramid_levels" << 1;
		fs << "template_pyramids" << "[";
		fs << "{";
		fs << "template_id" << 0; // 后面这里加循环写入模板增强
		fs << "template_width" << this->srcImg.cols;
		fs << "template_height" << this->srcImg.rows;

		fs << "features" << "[";
		for (int i = 0; i < this->out_features.size(); i++)
		{
			fs << "[:" << this->out_features[i].x << this->out_features[i].y <<
				this->out_features[i].label << "]";

		}
		fs << "]";
		fs << "}";
		fs << "]";
		fs.release();
	}

}


class shapeMatch
{
public:
	shapeMatch(cv::Mat in_testImg, float in_magnitude, float in_threshold, float in_iou, string in_feature_path);
	cv::Mat testImg;

	cv::Mat magnitudeImg;			// 梯度幅值图
	cv::Mat angle_ori;				// 角度图，0-360度
	cv::Mat quantized_angle;		// 量化（0-360  ->  0-7）后的方向
	cv::Mat spread_quantized;		// 广播后的方向
	float threshold;
	float magnitude;
	float iou;
	std::string feature_path;
	std::vector<std::vector<std::vector<Feature>>> in_features;
	std::vector<std::pair<int, int>> template_size;
	std::vector<cv::Mat> response_maps;

	int _T = 1;
	void load_shapeInfos();

	// 处理测试图片的梯度方向，广播，响应图
	void inference();
	void quantizedGradientOrientations();
	void spread(const Mat& src, Mat& dst, int T);

	void orUnaligned8u(const uchar* src, const int src_stride, uchar* dst, const int dst_stride,
		const int width, const int height);

	void computeResponseMaps(const Mat& src, std::vector<Mat>& in_response_maps);

	//void match();

};

shapeMatch::shapeMatch(cv::Mat in_testImg, float in_magnitude, float in_threshold, float in_iou, string in_feature_path)
{
	this->threshold = in_threshold;
	this->feature_path = in_feature_path;
	this->testImg = in_testImg;
	this->magnitude = in_magnitude;
	this->iou = in_iou;
	cout << "特征文件路径：" << this->feature_path << endl;
	cout << "输入的测试图梯度阈值：" << this->magnitude << endl;

	load_shapeInfos();

}


void shapeMatch::load_shapeInfos()
{
	cv::FileStorage fs(this->feature_path, cv::FileStorage::READ);
	FileNode fn = fs.root();
	vector<Feature> one_pyramid_features;
	FileNode tps_fn = fn["template_pyramids"];
	this->in_features.resize(tps_fn.size());
	this->template_size.resize(tps_fn.size());

	int expected_id = 0;
	FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
	for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
	{
		int template_id = (*tps_it)["template_id"];
		CV_Assert(template_id == expected_id);
		int _template_width = (*tps_it)["template_width"];
		int _template_height = (*tps_it)["template_height"];
		this->template_size[template_id].first = _template_width;
		this->template_size[template_id].second = _template_height;
		FileNode templates_fn = (*tps_it)["features"];
		one_pyramid_features.clear();
		one_pyramid_features.resize(templates_fn.size());

		FileNodeIterator templ_it = templates_fn.begin(), templ_it_end = templates_fn.end();
		int idx = 0;
		for (; templ_it != templ_it_end; ++templ_it)
		{
			Feature _one_feature;
			auto _feature_it = (*templ_it).begin();
			_feature_it >> _one_feature.x >> _one_feature.y >> _one_feature.label;
			one_pyramid_features[idx] = _one_feature;
			idx++;
		}

		this->in_features[expected_id].emplace_back(one_pyramid_features);
	}


}


void shapeMatch::quantizedGradientOrientations()//Mat &magnitude, Mat &quantized_angle, Mat &angle, float threshold
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	Mat_<unsigned char> quantized_unfiltered;
	this->angle_ori.convertTo(quantized_unfiltered, CV_8U, 16 / 360.0);

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
	for (int r = 1; r < this->angle_ori.rows - 1; ++r)
	{
		uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 1; c < this->angle_ori.cols - 1; ++c)
		{
			quant_r[c] &= 7;// 很巧妙地做了一个反转，如方向15的转为7,这里就是量化方向，360度换乘16块，再变成8块，也就是8个方向
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	this->quantized_angle = Mat::zeros(this->angle_ori.size(), CV_8U);
	float strong_magnitude_value = this->magnitude * this->magnitude;
	for (int r = 1; r < this->angle_ori.rows - 1; ++r)
	{
		float* mag_r = this->magnitudeImg.ptr<float>(r);

		for (int c = 1; c < this->angle_ori.cols - 1; ++c)
		{
			if (mag_r[c] > strong_magnitude_value)
			{
				// Compute histogram of quantized bins in 3x3 patch around pixel
				int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

				uchar* patch3x3_row = &quantized_unfiltered(r - 1, c - 1); // 太巧妙了
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
					this->quantized_angle.at<uchar>(r, c) = uchar(1 << index);
			}
		}
	}
}



void shapeMatch::inference()
{
	auto t0 = getTickCount();
	Mat smoothed;
	static const int KERNEL_SIZE = 3;
	cv::GaussianBlur(this->testImg, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

	cv::Mat sobel_dx, sobel_dy;
	cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
	cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
	this->magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
	cv::phase(sobel_dx, sobel_dy, this->angle_ori, true);
	// 量化到8个方向，再根据领域找出现次数最多的方向，主要输出结果是：this->quantized_angle
	quantizedGradientOrientations();
	spread(this->quantized_angle, this->spread_quantized, 8);		// 4 是广播的领域尺度，4 或 8

	computeResponseMaps(this->spread_quantized, this->response_maps);

	auto t1 = getTickCount();
	auto t01 = (t1 - t0) / getTickFrequency();
	cout << "测试图方向量化、广播、8个方向响应图耗时：" << t01 << endl;
	// 到这一步后基本需要的都可以了，开始匹配，输入：训练的特征点，8个响应图

	cv::Mat similarity = cv::Mat::zeros(this->testImg.size(), CV_32FC1);

	// 循环滑窗匹配
	{
		for (int p = 0; p < this->in_features.size(); p++) //这个循环是金字塔级别的
		{
			omp_set_num_threads(4);
#pragma omp parallel for
			for (int r = 0; r < quantized_angle.rows - this->template_size[p].second; r += _T)
			{

				for (int c = 0; c < quantized_angle.cols - this->template_size[p].first; c += _T)
				{
					for (int t = 0; t < this->in_features[p].size(); t++) // 这个循环是模板级别的
					{
						int fea_size = (int)this->in_features[p][t].size();
						int ori_sum = 0;
						// omp
						for (int f = 0; f < fea_size; f++) // 这个循环是特征级别的
						{
							Feature feat = this->in_features[p][t][f];
							int label = feat.label;
							auto _ori = (int)this->response_maps[label].ptr<uchar>(r + feat.y)[c + feat.x];
							/*if (_ori != 0)
							{
								cout << "label:" << label << ", x:" << r + feat.y << ", y:" << c + feat.x << ", ori: " <<
									 _ori << ", partial_sum: "<< ori_sum << endl;

							}*/
							ori_sum += (int)this->response_maps[label].ptr<uchar>(r + feat.y)[c + feat.x];

						}
						if (ori_sum != 0)
						{
							float score = ori_sum / (4.0f * fea_size);
							//cout << "fea_size: " << fea_size << ", ori_sum:" << ori_sum  << ", score:" << score << endl;
							similarity.at<float>(r, c) = score;

						}

					}
				}
			}
		}
	}

	// 筛选极大值
	class matchResult
	{
	public:
		int x;
		int y;
		float score;
		matchResult(int _x, int _y, float _score) :
			x(_x), y(_y), score(_score) {}
	};
	int _rows = similarity.rows;
	int _cols = similarity.cols;
	cv::Mat left = cv::Mat::zeros(similarity.size(), similarity.type());
	cv::Mat right = cv::Mat::zeros(similarity.size(), similarity.type());
	cv::Mat top = cv::Mat::zeros(similarity.size(), similarity.type());
	cv::Mat bottom = cv::Mat::zeros(similarity.size(), similarity.type());

	similarity.rowRange(0, _rows - 1).copyTo(top.rowRange(1, _rows));
	similarity.rowRange(1, _rows).copyTo(bottom.rowRange(0, _rows - 1));
	similarity.colRange(0, _cols - 1).copyTo(left.colRange(1, _cols));
	similarity.colRange(1, _cols).copyTo(right.colRange(0, _cols - 1));

	cv::Mat binary = similarity >= this->threshold
		& similarity >= left
		& similarity >= right
		& similarity >= top
		& similarity >= bottom;

	// 采用连通域的方法，有时会有两个位置连续的，也就是置信度一样的，这样应该可以避免
	cv::Mat labels, status, centroids;
	int label_count = cv::connectedComponentsWithStats(binary, labels, status, centroids);
	std::vector<matchResult> results;
	for (int i = 1; i < label_count; ++i)
	{
		int _x = status.at<int>(i, CC_STAT_LEFT);
		int _y = status.at<int>(i, CC_STAT_TOP);
		results.emplace_back(_x, _y, similarity.at<float>(_y, _x));
	}

	// 过一遍IOU，要求大于一定的比例就抛弃
	std::sort(results.begin(), results.end(), [&](matchResult a, matchResult b) {return a.score > b.score; });
	std::vector<int> del_results_idx;
	for (int n = 0; n < results.size(); n++)
	{
		cv::Rect box_0(results[n].x, results[n].y, this->template_size[0].first, this->template_size[0].second);
		for (int m = n + 1; m < results.size() - 1; m++)
		{
			cv::Rect box_1(results[m].x, results[m].y, this->template_size[0].first, this->template_size[0].second);
			float twoArea = 2 * this->template_size[0].first * this->template_size[0].second;
			//cv::Point iouBoxLeftUp(std::max(box_0.x, box_1.x), std::max(box_0.y, box_1.y));
			//cv::Point iouBoxRightLower(std::min(box_0.x + this->template_size[0].first, box_1.x), std::min(box_0.y, box_1.y) + this->template_size[0].second);
			int iouWidth = std::min(box_1.x + box_1.width, box_0.x + box_0.width) - std::max(box_0.x, box_1.x);
			int iouHeight = std::min(box_1.y + box_1.height, box_0.y + box_0.height) - std::max(box_0.y, box_1.y);
			float _iou = (iouWidth * iouHeight) / twoArea;

			if (_iou > this->iou)
			{
				del_results_idx.emplace_back(m);
			}
		}
	}
	std::set<int> s(del_results_idx.begin(), del_results_idx.end());
	del_results_idx.assign(s.begin(), s.end());

	for (int d = 0; d < del_results_idx.size(); d++)
	{
		results.erase(results.begin() + (del_results_idx.at(d) - d));

	}



	auto t21 = (getTickCount() - t1) / getTickFrequency();
	auto tall = (getTickCount() - t0) / getTickFrequency();
	cout << "滑窗匹配耗时：" << t21 << endl;
	cout << "形状匹配总耗时：" << tall << endl;

	// 可视化匹配结果的点
	cv::Mat show_img;
	cvtColor(this->testImg, show_img, COLOR_GRAY2RGB);
	for (int i = 0; i < results.size(); i++)
	{
		int offset_x = results[i].x;
		int offset_y = results[i].y;
		int feat_size = (int)this->in_features[0][0].size();
		for (int f = 0; f < feat_size; f++)
		{
			Feature feat = this->in_features[0][0][f];

			int x = offset_x + feat.x;
			int y = offset_y + feat.y;
			show_img.at<Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
		}
		putText(show_img, std::to_string(round(results[i].score * 1000) / 1000), cv::Point(results[i].x, results[i].y),
			1, 5, cv::Scalar(0, 255, 0), 2);

		cout << "匹配到第" << i << "个目标的置信度: " << results[i].score << endl;

	}

	int newWindowWidth, newWindowHeight, fixed = 1000;
	if (show_img.rows > show_img.cols)
	{
		float scale = (float)fixed / (float)show_img.rows;
		newWindowHeight = fixed;
		newWindowWidth = scale * show_img.cols;
	}
	else
	{
		float scale = (float)fixed / (float)show_img.cols;
		newWindowWidth = fixed;
		newWindowHeight = scale * show_img.rows;

	}
	cv::namedWindow("show_img", 0);
	cv::resizeWindow("show_img", cv::Size(newWindowWidth, newWindowHeight));
	cv::imshow("show_img", show_img);
	cv::waitKey(0);

	int a = 0;
}


static const unsigned char LUT3 = 3;
// 1,2-->0 3-->LUT3
CV_DECL_ALIGNED(16)
static const unsigned char SIMILARITY_LUT[256] = { 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, LUT3, 0, LUT3, 4, 4, LUT3, LUT3,
4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, LUT3, LUT3, LUT3,
LUT3, LUT3, LUT3, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 4, LUT3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0, LUT3, 4, 4, LUT3, LUT3, 4, 4, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, LUT3, LUT3, 4, 4, 4, 4, LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, LUT3, 0, 0, 0, 0,
LUT3, LUT3, LUT3, LUT3, 4, 4, 4, 4, 4, 4, 4, 4 };


void shapeMatch::computeResponseMaps(const Mat& src, std::vector<Mat>& in_response_maps)
{


	// Allocate response maps
	response_maps.resize(8);
	for (int i = 0; i < 8; ++i)
		response_maps[i].create(src.size(), CV_8U);

	Mat lsb4(src.size(), CV_8U);// 低位的四个方向，00001111
	Mat msb4(src.size(), CV_8U);// 高位的四个方向，11110000

	for (int r = 0; r < src.rows; ++r)
	{
		const uchar* src_r = src.ptr(r);
		uchar* lsb4_r = lsb4.ptr(r);
		uchar* msb4_r = msb4.ptr(r);

		for (int c = 0; c < src.cols; ++c)
		{
			// Least significant 4 bits of spread image pixel
			lsb4_r[c] = src_r[c] & 15;
			// Most significant 4 bits, right-shifted to be in [0, 16)
			msb4_r[c] = (src_r[c] & 240) >> 4;// 右移4位缩小到和低位一样的数值范围
		}
	}

	{
		uchar* lsb4_data = lsb4.ptr<uchar>();
		uchar* msb4_data = msb4.ptr<uchar>();

		// LUT is designed for 128 bits SIMD, so quite triky for others

		// For each of the 8 quantized orientations...
		for (int ori = 0; ori < 8; ++ori) {
			uchar* map_data = response_maps[ori].ptr<uchar>();
			const uchar* lut_low = SIMILARITY_LUT + 32 * ori;
			for (int i = 0; i < src.rows * src.cols; ++i)
			{
				// 查表，论文里面是通过不同方向求cos值，但这里不一样，用一个表（8个方向，每个方向有32种结果？），
				// 求最大响应来表示测试图的方向广播图在不同方向下的响应；结果已经预算好放到表中，直接读结果就行
				//
				// 广播后的一个像素方向根据8bit前后分两份，然后每一份有16种可能的方向，一个像素的前后两个方向 查找 表中 方向对应的响应，
				// 然后求两个方向最大的响应，得出该像素在8个方向中某个方向的响应值；
				//  
				map_data[i] = std::max(lut_low[lsb4_data[i]], lut_low[msb4_data[i] + 16]);


			}

		}


	}
}



void shapeMatch::spread(const Mat& src, Mat& dst, int T)
{
	// Allocate and zero-initialize spread (OR'ed) image
	dst = Mat::zeros(src.size(), CV_8U);

	// Fill in spread gradient image (section 2.3)
	for (int r = 0; r < T; ++r)
	{
		for (int c = 0; c < T; ++c)
		{
			orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
				static_cast<const int>(dst.step1()), src.cols - c, src.rows - r);
		}
	}
}


void shapeMatch::orUnaligned8u(const uchar* src, const int src_stride, uchar* dst, const int dst_stride,
	const int width, const int height)
{
	for (int r = 0; r < height; ++r)
	{
		int c = 0;

		for (; c < width; c++)
			dst[c] |= src[c];

		// Advance to next row
		src += src_stride;
		dst += dst_stride;
	}
}


void test_shape_match()
{
	string mode = "test";
	//string mode = "train";

	if (mode == "train")
	{
		cv::Mat template_img = cv::imread("D:\\1_industrial\\sheng_shape_match/干法/下半部分.png", 0);//规定输入的是灰度图，三通道的先不弄
		shapeInfoProducer trainer(template_img, 128, 50, "D:\\1_industrial\\sheng_shape_match/干法/train_template_lower.yaml");
		trainer.train();

	}
	else if (mode == "test")
	{
		cv::Mat test_img = cv::imread("D:\\1_industrial\\sheng_shape_match/干法/test.png", 0);	// sl_template_test     sl_test_4
		shapeMatch tester(test_img, 30, 0.93f, 0.01, "D:\\1_industrial\\sheng_shape_match/干法/train_template_up.yaml");
		tester.inference();

	}


}


void SL()
{
	cv::Mat test_img = cv::imread("D:\\1_industrial\\sheng_shape_match/sl_template.png", 0);

	Mat smoothed;
	static const int KERNEL_SIZE = 7;
	cv::GaussianBlur(test_img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

	cv::Mat sobel_dx, sobel_dy, magnitudeImg, angle_ori;
	cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
	cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
	magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
	cv::phase(sobel_dx, sobel_dy, angle_ori, true);
	//cv::Mat angle_ori_2;
	//phase(sobel_dy, sobel_dx, angle_ori_2, true);




}




void SL_2()
{
	//cv::Mat test_img = cv::imread("D:\\1_industrial\\sheng_shape_match/sl_template.png", 0);
	cv::String folder = "D:\\1_industrial\\sheng_shape_match\\test_2_v\\";
	//cv::String folder = "D:\\1_industrial\\sheng_shape_match\\cut\\";
	std::string hv = "v";
	//cv::String folder = "D:\\1_industrial\\sheng_shape_match\\h_img\\";
	//std::string hv = "h";
	std::vector<cv::String> imagePathList;
	cv::glob(folder, imagePathList);

	int sample_num = 3;							// 采样点的法向量方向前后各采样指定个数；
	float dist_threshold = 12.f;				// 两点的距离要大于一定的值才可以进行采样
	int sample_interval = 3;					// 长线段间隔采样,单位为像素
	for (int path_index = 0; path_index < imagePathList.size(); path_index++)
	{
		cv::Mat test_img = cv::imread(imagePathList[path_index], 0);

		cv::Mat show_img = test_img.clone();
		cvtColor(show_img, show_img, COLOR_GRAY2RGB);
		const int len = 205;
		////double org_x[len] = { 20.623625796954094,21.98200789953385, 211.47631120940991,215.5514575171492, 216.82217657321655, 172.90373769225675, 163.6577505594231,
		//		154.41176342658946, 144.32523200895275, 137.39074165932752, 136.55019737452446, 142.43400736814587, 152.52053878578258, 161.55638984741546,
		//		223.54653085164102, 230.06074905886473, 230.06074905886473, 222.70598656683796,12.990187508474849, 6.8962414436526736, 5.84556108764885,
		//		11.30909893886873, 61.53161995585148, 85.27699600153788, 92.63175849356465,97.675024202383, 101.03720134159524, 100.19665705679218,
		//		96.62434384637918, 91.58107813756082, 86.11754028634094, 70.9877431598859,65.94447745106754 };

		////double org_y[len] = { 71.03555228163074, 26.548538422143707, 26.548538422143707, 29.26530262730322, 72.04303282955664, 98.09990565845146, 94.94786459043999
		//	 , 90.32487102402317, 94.10732030563693, 102.09249101126599, 113.43983885610729 , 121.0047374193348, 125.83786705695239, 123.94664241614551
		//	 , 86.96269388481093, 79.60793139278417, 20.139423242967773, 12.574524679740247, 12.994796822141776 , 20.559695385369302, 75.40520996876887
		//	 , 84.86133317280328, 115.96147171051645, 125.2074588433501 , 122.89596206014168, 118.0628324225241, 112.80943064250499 , 105.0343960080767
		//	 , 98.09990565845146 , 93.6870481632354, 91.37555138002699, 98.09990565845146, 99.57085815685681 };


		double org_x[len] = { 144.08656311035156, 304.819580078125, 304.819580078125, 308.8641357421875, 312.8098449707031, 316.6409912109375, 320.3423767089844, 323.8995056152344, 327.2974853515625, 330.5204162597656, 333.55352783203125, 333.55242919921875, 365.96661376953125, 365.96661376953125, 365.96783447265625, 369.78509521484375, 373.17974853515625, 376.1140441894531, 378.5489196777344, 378.5406188964844, 413.0841369628906, 413.0841369628906, 413.09259033203125, 414.8156433105469, 416.080810546875, 416.8597412109375, 417.126220703125, 416.891845703125, 416.20465087890625, 415.087158203125, 413.56280517578125, 411.652587890625, 409.3809509277344, 406.7704162597656, 403.8433837890625, 400.6230773925781, 397.1322326660156, 393.393310546875, 389.4289855957031, 385.2629699707031, 380.9165954589844, 376.4135437011719, 371.7771911621094, 371.7771911621094, 304.00494384765625, 304.00494384765625, 304.0049743652344, 303.7215270996094, 299.4565124511719, 295.4841003417969, 291.8894958496094, 288.7574768066406, 286.17315673828125, 284.2216796875, 282.9893798828125, 282.5586853027344, 282.9893798828125, 284.2216796875, 286.17315673828125, 288.7574768066406, 291.8894958496094, 295.4841003417969, 299.4565124511719, 303.7215270996094, 307.7543640136719, 311.5312805175781, 314.9796142578125, 318.0284423828125, 318.0281982421875, 323.25469970703125, 329.166748046875, 332.34063720703125, 335.6404113769531, 339.049560546875, 342.5541687011719, 342.5541687011719, 371.7771911621094, 371.7771911621094, 375.4326477050781, 378.83795166015625, 381.91937255859375, 384.60406494140625, 386.8189697265625, 388.49139404296875, 389.54791259765625, 389.91650390625, 389.4985656738281, 388.3039855957031, 388.3002624511719, 388.3002624511719, 353.7563781738281, 353.7597351074219, 351.6121520996094, 348.727294921875, 348.72711181640625, 348.72711181640625, 316.31292724609375, 316.31329345703125, 313.810791015625, 311.0288391113281, 308.0157165527344, 304.819580078125, 304.819580078125, 144.08656311035156, 144.08656311035156, 140.890380859375, 137.8771209716797, 135.09535217285156, 132.59278869628906, 132.59315490722656, 132.59315490722656, 100.17892456054688, 100.17863464355469, 97.29438781738281, 95.14708709716797, 95.1494369506836, 95.1494369506836, 60.60572052001953, 60.603302001953125, 59.40739059448242, 58.989013671875, 59.3575439453125, 60.4145622253418, 62.086997985839844, 64.30201721191406, 66.98666381835938, 70.06792449951172, 73.47279357910156, 77.12850952148438, 77.12850952148438, 106.35210418701172, 106.35210418701172, 109.85598754882812, 113.26560974121094, 116.56507873535156, 119.73963165283203, 125.6511001586914, 130.87786865234375, 130.87777709960938, 130.87777709960938, 130.877685546875, 133.9263458251953, 137.37498474121094, 141.1513671875, 145.1843719482422, 149.44935607910156, 153.4217987060547, 157.01661682128906, 160.14874267578125, 162.73301696777344, 164.6841583251953, 165.9173583984375, 166.3470001220703, 165.9173583984375, 164.6841583251953, 162.73301696777344, 160.14874267578125, 157.01661682128906, 153.4217987060547, 149.44935607910156, 145.1843719482422, 144.89932250976562, 144.9012451171875, 77.12850952148438, 77.12850952148438, 72.4919204711914, 67.98919677734375, 63.643218994140625, 59.47675704956055, 55.51264953613281, 51.77360153198242, 48.282508850097656, 45.06208801269531, 42.13517761230469, 39.524574279785156, 37.2530403137207, 35.34343719482422, 33.81852722167969, 32.70116424560547, 32.013851165771484, 31.779699325561523, 32.04578399658203, 32.82549285888672, 34.09144973754883, 35.81560516357422, 35.821380615234375, 70.36512756347656, 70.36512756347656, 70.35928344726562, 72.79399871826172, 75.7276382446289, 79.12212371826172, 82.938232421875, 82.93912506103516, 115.35327911376953, 115.35327911376953, 115.35238647460938, 118.38570404052734, 121.60863494873047, 125.00626373291016, 128.5633087158203, 132.26507568359375, 136.09616088867188, 140.04164123535156, 144.08656311035156 };
		double org_y[len] = { 209.65289306640625, 209.65289306640625, 209.65289306640625, 209.47731018066406, 208.96087646484375, 208.11834716796875, 206.9646453857422, 205.51478576660156, 203.7829132080078, 201.78440856933594, 199.5341796875, 199.53341674804688, 173.3621368408203, 173.3621368408203, 173.36300659179688, 169.90139770507812, 166.03378295898438, 161.79798889160156, 157.23233032226562, 157.22364807128906, 82.05640411376953, 82.05640411376953, 82.0650405883789, 77.73250579833984, 73.19355773925781, 68.47571563720703, 63.60598373413086, 59.03496551513672, 54.59579086303711, 50.311214447021484, 46.20361328125, 42.29551696777344, 38.6092643737793, 35.16755294799805, 31.992589950561523, 29.107074737548828, 26.533353805541992, 24.293882369995117, 22.411287307739258, 20.907936096191406, 19.806243896484375, 19.128786087036133, 18.897903442382812, 18.897903442382812, 18.897903442382812, 18.897903442382812, 18.897878646850586, 18.8961238861084, 19.319929122924805, 20.53562355041504, 22.459228515625, 25.006906509399414, 28.09470558166504, 31.638681411743164, 35.555084228515625, 39.759864807128906, 43.964630126953125, 47.88092803955078, 51.425010681152344, 54.51276779174805, 57.0604362487793, 58.98408508300781, 60.19987106323242, 60.62370681762695, 60.24531936645508, 59.15715789794922, 57.42979049682617, 55.13395309448242, 55.13370132446289, 51.197914123535156, 48.2368278503418, 47.15965270996094, 46.371437072753906, 45.8874626159668, 45.722808837890625, 45.722808837890625, 45.722808837890625, 45.722808837890625, 46.08612823486328, 47.12813186645508, 48.77692794799805, 50.96064758300781, 53.60737228393555, 56.64508056640625, 60.00199890136719, 63.60598373413086, 67.44107818603516, 70.98968505859375, 70.98625183105469, 70.98625183105469, 146.15357971191406, 146.156982421875, 149.67752075195312, 152.60916137695312, 152.60887145996094, 152.60887145996094, 178.78001403808594, 178.78021240234375, 180.47982788085938, 181.7524871826172, 182.55108642578125, 182.8279266357422, 182.8279266357422, 182.8279266357422, 182.8279266357422, 182.55108642578125, 181.75250244140625, 180.4798583984375, 178.7803497314453, 178.78001403808594, 178.78001403808594, 152.60887145996094, 152.6092529296875, 149.67835998535156, 146.15896606445312, 146.15357971191406, 146.15357971191406, 70.98625183105469, 70.99162292480469, 67.44214630126953, 63.60598373413086, 60.00199890136719, 56.64508056640625, 53.60737228393555, 50.96064758300781, 48.77692794799805, 47.12813186645508, 46.08612823486328, 45.722808837890625, 45.722808837890625, 45.722808837890625, 45.722808837890625, 45.8874626159668, 46.371437072753906, 47.159645080566406, 48.2368049621582, 51.19789505004883, 55.133663177490234, 55.13385009765625, 55.13385009765625, 55.133975982666016, 57.429813385009766, 59.15715789794922, 60.24531936645508, 60.62370681762695, 60.19987106323242, 58.98408508300781, 57.0604362487793, 54.51276779174805, 51.425010681152344, 47.88092803955078, 43.964630126953125, 39.759864807128906, 35.555084228515625, 31.638681411743164, 28.09470558166504, 25.006906509399414, 22.459228515625, 20.53562355041504, 19.319929122924805, 18.8961238861084, 18.897891998291016, 18.897903442382812, 18.897903442382812, 18.897903442382812, 19.128786087036133, 19.806243896484375, 20.907936096191406, 22.411287307739258, 24.293882369995117, 26.533353805541992, 29.107074737548828, 31.992589950561523, 35.16755294799805, 38.6092643737793, 42.29551696777344, 46.20361328125, 50.311214447021484, 54.59579086303711, 59.03496551513672, 63.60598373413086, 68.4771499633789, 73.1963882446289, 77.73646545410156, 82.07012176513672, 82.05640411376953, 157.22364807128906, 157.22364807128906, 157.2373504638672, 161.80174255371094, 166.0362091064453, 169.90255737304688, 173.3631134033203, 173.3621368408203, 199.53341674804688, 199.53341674804688, 199.53443908691406, 201.78448486328125, 203.78298950195312, 205.5148162841797, 206.9646759033203, 208.11834716796875, 208.9608917236328, 209.47731018066406, 209.65289306640625 };

		float templateImg_width = test_img.cols, templateImg_height = test_img.rows;

		std::vector<float> x, y;
		if (hv == "h")
		{
			for (int idx = 0; idx < len; idx++)
			{
				y.emplace_back(org_y[idx]);
				x.emplace_back(org_x[idx]);
			}
		}
		else if (hv == "v")
		{
			for (int idx = 0; idx < len; idx++)
			{
				y.emplace_back(templateImg_height - org_y[idx]);
				x.emplace_back(org_x[idx]);
			}
		}
		else
		{
			cout << "不匹配的参数：h or v" << endl;
		}

		bool is_line = false;
		//bool is_line = true;

		// 画线
		if (is_line)
		{
			for (int i = 0; i < 32; i++)
			{
				cv::Point2f c_1 = cv::Point2f(x[i], y[i]);
				cv::Point2f c_2 = cv::Point2f(x[i + 1], y[i + 1]);
				cv::line(show_img, c_1, c_2, cv::Scalar(0, 255, 0), 1);
			}
			// LAST line
			cv::Point2f c_1 = cv::Point2f(x[0], y[0]);
			cv::Point2f c_2 = cv::Point2f(x[32], y[32]);
			cv::line(show_img, c_1, c_2, cv::Scalar(0, 255, 0), 1);

		}
		else
		{
			//画点
			for (int i = 0; i < len; i++)
			{
				cv::Point2f c_1(x[i], y[i]);
				//cv::circle(show_img, c_1, 1, cv::Scalar(0, 0, 255));
				//cv::putText(show_img, std::to_string(i), c_1, 1, 0.4, cv::Scalar(0, 240, 0));
			}

		}

		// 顺络框偏移思路：1.每一个点的法向量（垂直）上（一定的领域内）找梯度最大的点，即为模板点对应在测试图片上的点

		auto t_1 = getTickCount();
		Mat smoothed;
		static const int KERNEL_SIZE = 7;
		cv::GaussianBlur(test_img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

		cv::Mat sobel_dx, sobel_dy, magnitudeImg, angle_ori;
		cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
		//cv::phase(sobel_dx, sobel_dy, angle_ori, true);
		//cv::Mat angle_ori_2;
		//cv::phase(sobel_dy, sobel_dx, angle_ori_2, true);
		auto t_2 = getTickCount();
		cout << "梯度提取耗时：" << (t_2 - t_1) / cv::getTickFrequency() << endl;

		//0225新思路：
		// 1.判断两点的线是否大于一定的距离，大于就在直线上的线段采样
		// 2.判断斜率，有些斜率是水平或垂直的，得指定k, b，
		// 
		// 
		// 两点为线，求这根线第一点的法向量，也就是求K值斜率
		std::vector<cv::Point2f> org_points, dst_points;
		int64 t_3;
		//#pragma omp parallel for
		//#pragma omp parallel num_threads(2)
		for (int i = 0; i < len; i++)
		{
			//cout << "模板点序号: " << i << endl;
			cv::Point2f c_1, c_2;
			// 下面这个判断是为了把全部点都用起来，第一个和最后一个
			if (i < len - 1)
			{
				c_1 = cv::Point2f(x[i], y[i]);
				c_2 = cv::Point2f(x[i + 1], y[i + 1]);
			}
			else if (i == len - 1)
			{
				c_1 = cv::Point2f(x[len - 1], y[len - 1]);
				c_2 = cv::Point2f(x[0], y[0]);
			}
			// 确保x1在上方，即c_1.y > c_2.y
			if (c_1.y > c_2.y)
			{
				auto _temp_c_2 = c_2;
				c_2 = c_1;
				c_1 = _temp_c_2;
			}

			// 先判断距离
			cv::circle(show_img, c_1, 1, cv::Scalar(0, 0, 255));
			cv::circle(show_img, c_2, 1, cv::Scalar(0, 0, 255));
			float point_dist = std::sqrt((c_2.x - c_1.x) * (c_2.x - c_1.x) + (c_2.y - c_1.y) * (c_2.y - c_1.y));

			if (point_dist > dist_threshold)
			{
				float fenzi = c_2.y - c_1.y;
				float fenmu = c_2.x - c_1.x;
				//cout << "K: " << k << endl;
				if (abs(fenmu) < 0.6)
				{
					cout << "垂直线段" << endl;
					// 原本是垂直的线段，其法向量就是水平的，采样点就是水平的，x方向不变,y方向变
					for (int sample_idx = 1; sample_idx < point_dist / sample_interval; sample_idx++)
					{
						float _y = c_1.y + sample_idx * sample_interval;

						std::vector<cv::Point2f> sample_points;		// 采样点,一次采样点出一个结果，加到org_points 和dst_points
						std::vector<float> sample_mag;				// 采样点对应的梯度幅值
						for (int sx = 0; sx < sample_num; sx++)
						{
							float x1 = c_1.x - sx;
							float x2 = c_1.x + sx;

							if (0 <= x1 && x1 < magnitudeImg.cols)
							{
								cv::Point2f _sp(x1, _y);
								sample_points.emplace_back(_sp);
								sample_mag.emplace_back(magnitudeImg.at<float>(_y, x1));
								show_img.at<Vec3b>(_y, x1) = cv::Vec3b(180, 0, 0);
							}
							if (0 <= x2 && x2 < magnitudeImg.cols)
							{
								cv::Point2f _sp(x2, _y);
								sample_points.emplace_back(_sp);
								sample_mag.emplace_back(magnitudeImg.at<float>(_y, x2));
								show_img.at<Vec3b>(_y, x2) = cv::Vec3b(180, 0, 0);
							}
						}
						int max_idx = std::max_element(sample_mag.begin(), sample_mag.end()) - sample_mag.begin();
						org_points.emplace_back(sample_points[0]);
						dst_points.emplace_back(sample_points[max_idx]);
						show_img.at<Vec3b>(sample_points[0].y, sample_points[0].x) = cv::Vec3b(255, 0, 255);
						show_img.at<Vec3b>(sample_points[max_idx].y, sample_points[max_idx].x) = cv::Vec3b(255, 255, 255);
					}

				}
				else if (abs(fenzi) < 0.6)
				{
					cout << "水平线段" << endl;
					// 原本是水平的线段，其法向量就是垂直，采样点就是垂直的，y方向不变
					int sample_base_num = point_dist > sample_interval ? point_dist / sample_interval : 1;
					//int new_sample_inter
					for (int sample_idx = 1; sample_idx <= sample_base_num; sample_idx++)
					{
						float add_or_sub = c_1.x > c_2.x ? -1.0f : 1.0f;
						float _x = c_1.x + sample_idx * sample_interval * add_or_sub;

						std::vector<cv::Point2f> sample_points;		// 采样点,一次采样点出一个结果，加到org_points 和dst_points
						std::vector<float> sample_mag;				// 采样点对应的梯度幅值
						for (int sx = 0; sx < sample_num; sx++)
						{
							float y1 = c_1.y - sx;
							float y2 = c_1.y + sx;

							if (0 <= y1 && y1 < magnitudeImg.rows)
							{
								cv::Point2f _sp(_x, y1);
								sample_points.emplace_back(_sp);
								sample_mag.emplace_back(magnitudeImg.at<float>(y1, _x));
								show_img.at<Vec3b>(y1, _x) = cv::Vec3b(180, 0, 0);
							}
							if (0 <= y2 && y2 < magnitudeImg.rows)
							{
								cv::Point2f _sp(_x, y2);
								sample_points.emplace_back(_sp);
								sample_mag.emplace_back(magnitudeImg.at<float>(y2, _x));
								show_img.at<Vec3b>(y2, _x) = cv::Vec3b(180, 0, 0);
							}
						}
						int max_idx = std::max_element(sample_mag.begin(), sample_mag.end()) - sample_mag.begin();
						org_points.emplace_back(sample_points[0]);
						dst_points.emplace_back(sample_points[max_idx]);
						show_img.at<Vec3b>(sample_points[0].y, sample_points[0].x) = cv::Vec3b(255, 0, 255);
						show_img.at<Vec3b>(sample_points[max_idx].y, sample_points[max_idx].x) = cv::Vec3b(255, 255, 255);
					}

				}
				else
				{
					// 正常的倾斜线段，根据斜率k和截距b 获得新的坐标
					float k = fenzi / fenmu;
					float b = c_1.y - c_1.x * k;	// 需要原始的向量参数：斜率k，截距b，去计算新的采样基准点；前面特殊的垂直水平情况不需要，直接固定x或y坐标
					// 两条垂直相交直线的斜率相乘积为-1；
					//cout << "正常线段, k: "<< k << endl;
					float new_k = -1 / k;

					// 确定如何制作采样点：线段有偏横和偏竖着的，偏横的用x坐标采样，偏竖着用y坐标采样
					int sample_base_num = point_dist > sample_interval ? point_dist / sample_interval : point_dist / 2;
					float x_dist = std::abs(c_1.x - c_2.x);
					float y_dist = std::abs(c_1.y - c_2.y);
					std::string statu = x_dist > y_dist ? "x" : "y";
					float x_or_y_interval = x_dist > y_dist ? x_dist / sample_base_num : y_dist / sample_base_num;
					for (int sample_idx = 1; sample_idx <= sample_base_num; sample_idx++)
					{
						// 先确立采样基准点，根据第一次的k,b
						float _x, _y;
						x_or_y_interval = sample_base_num == 1 ? x_or_y_interval / 2 : x_or_y_interval;

						if (statu == "x")
						{
							// 取小的是因为出现这种情况：c_2在左下方，c_1在右上方，直接拿c_1
							_x = std::min(c_1.x, c_2.x) + sample_idx * x_or_y_interval;
							_y = _x * k + b;
						}
						else if (statu == "y")
						{
							_y = c_1.y + sample_idx * x_or_y_interval;
							_x = (_y - b) / k;
						}
						//show_img.at<Vec3b>(_y, _x) = cv::Vec3b(255, 0, 0);
						float new_b = _y - _x * new_k;		// 采样基准点变了会导致截距变，但k不变

						std::vector<cv::Point2f> sample_points;		// 采样点,一次采样点出一个结果，加到org_points 和dst_points
						std::vector<double> sample_mag;				// 采样点对应的梯度幅值
						std::vector<cv::Point2f> vector_xy;
						cv::Point2f base_sample_vector_xy(sobel_dx.at<float>(_y, _x), sobel_dy.at<float>(_y, _x));
						for (int sx = 0; sx < sample_num; sx++)
						{
							float x1 = _x - sx;
							float y1 = x1 * new_k + new_b;

							float x2 = _x + sx;
							float y2 = x2 * new_k + new_b;

							if ((0 <= y1 && y1 < magnitudeImg.rows) && (0 <= x1 && x1 < magnitudeImg.cols))
							{
								cv::Point2f _sp(x1, y1);
								// 添加向量
								cv::Point2f _sample_vector_xy(sobel_dx.at<float>(y1, x1), sobel_dy.at<float>(y1, x1));
								vector_xy.emplace_back(_sample_vector_xy);

								sample_points.emplace_back(_sp);
								sample_mag.emplace_back(magnitudeImg.at<float>(y1, x1));
								show_img.at<Vec3b>(y1, x1) = cv::Vec3b(180, 0, 0);
							}
							if (0 <= y2 && y2 < magnitudeImg.rows && (0 <= x2 && x2 < magnitudeImg.cols))
							{
								cv::Point2f _sp(x2, y2);
								// 添加向量
								cv::Point2f _sample_vector_xy(sobel_dx.at<float>(y2, x2), sobel_dy.at<float>(y2, x2));
								vector_xy.emplace_back(_sample_vector_xy);

								sample_points.emplace_back(_sp);
								sample_mag.emplace_back(magnitudeImg.at<float>(y2, x2));
								show_img.at<Vec3b>(y2, x2) = cv::Vec3b(180, 0, 0);
							}
						}
						// 用基准点的向量和采样点的所有向量点乘内积，求最大的
						std::vector<float> inner;
						for (auto& s : vector_xy)
						{
							float _inner = base_sample_vector_xy.x * s.x + base_sample_vector_xy.y * s.y;
							inner.emplace_back(_inner);
						}

						int max_idx = std::max_element(inner.begin(), inner.end()) - inner.begin();
						//int max_idx = std::max_element(sample_mag.begin(), sample_mag.end()) - sample_mag.begin();
						org_points.emplace_back(sample_points[0]);
						dst_points.emplace_back(sample_points[max_idx]);
						show_img.at<Vec3b>(sample_points[0].y, sample_points[0].x) = cv::Vec3b(255, 0, 255);
						show_img.at<Vec3b>(sample_points[max_idx].y, sample_points[max_idx].x) = cv::Vec3b(255, 255, 255);
					}
				}

			}
			else
			{
				//cout << "两点距离不满足采样条件，跳过，两点距离："<< point_dist << endl;
				continue;

			}

		}

		std::vector<cv::Point2f> org_xy;
		for (int i = 0; i < len; i++)
		{
			org_xy.emplace_back(x[i], y[i]);
		}


		t_3 = getTickCount();
		cout << "提取法向量采样基准点耗时：" << (t_3 - t_2) / cv::getTickFrequency() << endl;


		// 透视变换
		//org_points.insert(org_points.end(), org_xy.begin(), org_xy.end());
		//dst_points.insert(dst_points.end(), org_xy.begin(), org_xy.end());
		//Mat m = findHomography(org_points, dst_points, 16, 0);	// RANSAC
		cv::Mat m;
		for (int f = 0; f < 100; f++)
		{
			auto t_f0 = getTickCount();
			m = findHomography(org_points, dst_points, 8);	// RANSAC
			auto t_f1 = getTickCount();
			cout << f << " , 计算单应性变换矩阵耗时：" << (t_f1 - t_f0) / getTickFrequency() << endl;
		}
		std::vector<cv::Point2f> trans_points;
		trans_points.resize(org_xy.size());
		perspectiveTransform(org_xy, trans_points, m);
		auto t_4 = getTickCount();
		cout << "透视变换耗时：" << (t_4 - t_3) / cv::getTickFrequency() << endl;

		auto t_cost = (getTickCount() - t_1) / getTickFrequency();
		cout << "----------------------- 校正一次耗时：" << t_cost << endl;
		cv::Mat show_img_2, show_img_3;
		cvtColor(test_img, show_img_2, COLOR_GRAY2BGR);
		show_img_3 = show_img_2.clone();
		{
			for (int i = 0; i < trans_points.size() - 1; i++)
			{
				//画透视变换后的
				cv::line(show_img_2, trans_points[i], trans_points[i + 1], cv::Scalar(0, 255, 0), 1);
				//画原始的
				cv::line(show_img_3, org_xy[i], org_xy[i + 1], cv::Scalar(0, 255, 255), 1);
			}
			// LAST line， 透视后的
			cv::line(show_img_2, trans_points[0], trans_points[trans_points.size() - 1], cv::Scalar(0, 255, 0), 1);
			// LAST line 原始
			cv::line(show_img_3, org_xy[0], org_xy[org_xy.size() - 1], cv::Scalar(0, 255, 255), 1);

		}

	}

}

//int count_sobel(uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e, uint8_t f) {
//
//	return (((int)a + 2 * (int)b + (int)c) - ((int)d + 2 * (int)e + (int)f));
//
//}


//核函数
void sobel_gpu_fun(int pix, uint8_t* img_in, float* sobelx, float* sobely, float* sobel_out, int img_w, int img_h)
{
	//int row = blockDim.y * blockIdx.y + threadIdx.y;
	//int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = pix / img_w;
	int col = pix % img_w;
	auto count_sobel = [=](uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint8_t e, uint8_t f)
	{
		return (((int)a + 2 * (int)b + (int)c) - ((int)d + 2 * (int)e + (int)f));
	};
	if ((row >= 1) && (row < (img_h - 1)) && (col >= 1) && (col < (img_w - 1)))
	{
		uint8_t x1, x2, x3, x4, x6, x7, x8, x9;
		x1 = img_in[(row - 1) * img_w + (col - 1)];
		x2 = img_in[(row - 1) * img_w + col];
		x3 = img_in[(row - 1) * img_w + col + 1];
		x4 = img_in[row * img_w + col - 1];
		x6 = img_in[row * img_w + col + 1];
		x7 = img_in[(row + 1) * img_w + col - 1];
		x8 = img_in[(row + 1) * img_w + col];
		x9 = img_in[(row + 1) * img_w + col + 1];

		float dfdx = count_sobel(x1, x4, x7, x3, x6, x9);
		float dfdy = count_sobel(x1, x2, x3, x7, x8, x9);
		float gradient = sqrtf(dfdy * dfdy + dfdx * dfdx);

		sobelx[row * img_w + col] = (float)dfdx;
		sobely[row * img_w + col] = (float)dfdy;
		sobel_out[row * img_w + col] = (float)gradient;
	}
}


//double gaussFunc1D(int x, double sigma)
//{
//	double A = 1.0 / (sigma * sqrt(2 * 3.141592653));
//	double index = -1.0 * ((double)x * (double)x) / (2 * (double)sigma * (double)sigma);
//	return A * exp(index);
//}


void getKernal(double* weight)
{
	int radius = 1;
	double sum = 0;
	double sigma = 1.0f;
	auto gaussFunc1D = [=](int x, double sigma)
	{
		double A = 1.0 / (sigma * sqrt(2 * 3.141592653));
		double index = -1.0 * ((double)x * (double)x) / (2 * (double)sigma * (double)sigma);
		return A * exp(index);
	};
	// 获取权值空间weight[]
	for (int i = 0; i < 2 * radius + 1; i++)
	{
		weight[i] = gaussFunc1D(i - radius, sigma);
		sum += weight[i];
	}
	// 归一化
	for (int i = 0; i < 2 * radius + 1; i++)
	{
		weight[i] /= sum;
	}
}



void simulation_cuda_sobel()
{

	cv::String folder = "D:\\1_industrial\\sheng_shape_match\\test_2_v\\";
	std::vector<cv::String> imagePathList;
	cv::glob(folder, imagePathList);
	// 高斯模糊矩阵 ：
	//float guassMatrix[3][3] = { {0.0947416, 0.118318, 0.0947416},{0.118318, 0.147761, 0.118318},{0.0947416, 0.118318, 0.0947416} };


	for (int path_index = 0; path_index < imagePathList.size(); path_index++)
	{
		cv::Mat test_img = cv::imread(imagePathList[path_index], 0);

		//Mat smoothed;
		//static const int KERNEL_SIZE = 3;
		//cv::GaussianBlur(test_img, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
		//cv::Mat sobel_dx, sobel_dy, magnitudeImg, angle_ori;
		//cv::Sobel(test_img, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		//cv::Sobel(test_img, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		//magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);


		// 模拟cuda 高斯模糊 -> sobel -> 根据轮廓找点 -> 透视变换矩阵M
		int gaussKernelSize = 3;
		cv::Mat selfSmoothed = test_img.clone();

		auto t_0 = getTickCount();
		// 高斯模糊
		{
			auto gaussWeight = std::make_unique<double[]>(gaussKernelSize);
			getKernal(gaussWeight.get());
			const int byteLen = 1;
			// 在横向进行一次相加
			for (int y = 0; y < test_img.rows; y++)
			{
				for (int x = 1; x < test_img.cols - 1; x++)
				{
					double newPix = 0.0;
					//// 边界处理后的对应的权值矩阵实际值
					newPix += (float)test_img.data[y * test_img.step[0] + (x - 1) * byteLen] * gaussWeight[0];
					newPix += (float)test_img.data[y * test_img.step[0] + x * byteLen] * gaussWeight[1];
					newPix += (float)test_img.data[y * test_img.step[0] + (x + 1) * byteLen] * gaussWeight[2];
					selfSmoothed.data[y * selfSmoothed.step[0] + x * byteLen] = (uchar)newPix;
				}
			}
			// 在竖向进行一次相加
			for (int y = 1; y < test_img.rows - 1; y++)
			{
				for (int x = 0; x < test_img.cols; x++)
				{
					double newPix = 0.0f;
					//// 边界处理后的对应的权值矩阵实际值
					newPix += (float)selfSmoothed.data[(y - 1) * selfSmoothed.step[0] + x * byteLen] * gaussWeight[0];
					newPix += (float)selfSmoothed.data[y * selfSmoothed.step[0] + x * byteLen] * gaussWeight[1];
					newPix += (float)selfSmoothed.data[(y + 1) * selfSmoothed.step[0] + x * byteLen] * gaussWeight[2];
					selfSmoothed.data[y * selfSmoothed.step[0] + x * byteLen] = (uchar)newPix;
				}
			}
		}

		auto t_1 = getTickCount();
		cout << "手动计算高斯模糊耗时：" << (t_1 - t_0) / getTickFrequency() << endl;

		cv::Mat cuda_sobel(test_img.size(), CV_32FC1, cv::Scalar(0));
		cv::Mat cuda_sobelx(test_img.size(), CV_32FC1, cv::Scalar(0));
		cv::Mat cuda_sobely(test_img.size(), CV_32FC1, cv::Scalar(0));
		// sobel梯度计算
		{
			for (int pix = 0; pix < test_img.cols * test_img.rows; pix++)
			{
				auto imgData = selfSmoothed.data;
				sobel_gpu_fun(pix, selfSmoothed.data, (float*)cuda_sobelx.data, (float*)cuda_sobely.data, (float*)cuda_sobel.data, selfSmoothed.cols, selfSmoothed.rows);

			}
		}
		auto t_2 = getTickCount();
		cout << "手动计算sobel梯度耗时：" << (t_2 - t_1) / getTickFrequency() << endl;


		// 找匹配点



		//  求透视变换矩阵

		int out = 9;
	}
}


void showKernel()
{

	double sigma = 1;
	double error = 0.001;
	double pi = 3.141592653;

	// 根据signma 生成卷积
	// 1. 计算需要的卷积核大小。
	//double kernel_size = std::ceil(std::sqrt(std::log(std::sqrt(2 * pi) * sigma * error) * -1 * 2 * sigma * sigma));
	double kernel_size = 5;
	int k_w = int(2 * kernel_size + 1);
	int k_h = int(2 * kernel_size + 1);
	cout << "二阶特征值卷积核心: w=" << k_w << ", h=" << k_h << endl;

	// 2. 计算需要的卷积核
	// 参考资料： 
	// [1] https://cds.cern.ch/record/400314/files/p27.pdf
	// [2] Steger, Carsten. "An unbiased detector of curvilinear structures." IEEE Transactions on pattern analysis and machine intelligence 20.2 (1998): 113-125.
	// 我们需要计算某个点附近的二阶导数，为了处理大线宽的情况，直接使用Sobel算子之类需要先对图像过一次高斯模糊，再使用两次Sobel算子分别计算一阶导数和二阶导数，
	// 也就是说为了得到二阶导数我们需要做三次卷积，这相当不划算。而使用高斯函数的导数生成卷积核可以通过一次卷积得到二阶导数。
	auto g_sigma1 = [=](double _x, double _y, double _sigma) {
		return 1.0 / std::sqrt((2.0 * pi * _sigma * _sigma)) * std::exp(-1.0 / 2.0 * (_x * _x / _sigma / _sigma + _y * _y / _sigma / _sigma));
	};
	auto g_sigma1_x = [=](double _x, double _y, double _sigma) {
		return -1.0 / (_sigma * _sigma) * _x * g_sigma1(_x, _y, _sigma);
	};
	auto g_sigma1_y = [=](double _x, double _y, double _sigma) {
		return -1.0 / (_sigma * _sigma) * _y * g_sigma1(_x, _y, _sigma);
	};
	cv::Mat kernel_dxx = cv::Mat::zeros(k_h, k_w, CV_32SC1);
	cv::Mat kernel_dxy = cv::Mat::zeros(k_h, k_w, CV_32SC1);
	cv::Mat kernel_dyy = cv::Mat::zeros(k_h, k_w, CV_32SC1);
	const int Q_BIT = 16; // 整数计算性能通常会远高于浮点计算性能，我们需要先对卷积核做量化。
	for (int m = 0; m < k_h; ++m) {
		for (int n = 0; n < k_w; ++n) {
			double y = m - (k_h - 1.0) / 2.0;
			double x = n - (k_w - 1.0) / 2.0;
			double dxx = g_sigma1_x(x + 0.5, y, sigma) - g_sigma1_x(x - 0.5, y, sigma);
			double dxy = g_sigma1_x(x, y + 0.5, sigma) - g_sigma1_x(x, y - 0.5, sigma);
			double dyy = g_sigma1_y(x, y + 0.5, sigma) - g_sigma1_y(x, y - 0.5, sigma);
			int32_t Q_dxx = static_cast<int32_t>(std::round(dxx * (1ULL << Q_BIT)));
			int32_t Q_dxy = static_cast<int32_t>(std::round(dxy * (1ULL << Q_BIT)));
			int32_t Q_dyy = static_cast<int32_t>(std::round(dyy * (1ULL << Q_BIT)));
			kernel_dxx.at<int32_t>(m, n) = Q_dxx;
			kernel_dxy.at<int32_t>(m, n) = Q_dxy;
			kernel_dyy.at<int32_t>(m, n) = Q_dyy;
		}
	}
	cv::String folder = "D:\\1_industrial\\sheng_shape_match\\test_2_v\\";
	std::vector<cv::String> imagePathList;
	cv::glob(folder, imagePathList);
	// 高斯模糊矩阵 ：
	float guassMatrix[3][3] = { {0.0947416, 0.118318, 0.0947416},{0.118318, 0.147761, 0.118318},{0.0947416, 0.118318, 0.0947416} };


	for (int path_index = 0; path_index < imagePathList.size(); path_index++)
	{
		cv::Mat test_img = cv::imread(imagePathList[path_index], 0);
		cv::Mat dst_dx, dst_dxy, dst_dyy, magnitudeImg;
		cv::filter2D(test_img, dst_dx, CV_32F, kernel_dxx);
		cv::filter2D(test_img, dst_dxy, CV_32F, kernel_dxy);
		cv::filter2D(test_img, dst_dyy, CV_32F, kernel_dyy);

		magnitudeImg = dst_dx.mul(dst_dx) + dst_dyy.mul(dst_dyy) + dst_dxy.mul(dst_dxy);

		magnitudeImg.convertTo(magnitudeImg, CV_32F, 1.0 / (1 << 16));


	}


}



void simulationGuass()
{

	cv::String folder = "D:\\1_industrial\\sheng_shape_match\\test_2_v\\";
	std::vector<cv::String> imagePathList;
	cv::glob(folder, imagePathList);
	// 高斯模糊矩阵 ：
	float guassMatrix[3][3] = { {0.0947416, 0.118318, 0.0947416},{0.118318, 0.147761, 0.118318},{0.0947416, 0.118318, 0.0947416} };


	for (int path_index = 0; path_index < imagePathList.size(); path_index++)
	{
		cv::Mat test_img = cv::imread(imagePathList[path_index], 0);
		cv::Mat cvSmoothed;
		static const int KERNEL_SIZE = 3;
		auto t_00 = getTickCount();
		cv::GaussianBlur(test_img, cvSmoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

		auto t_01 = getTickCount();
		cout << "cv计算高斯模糊耗时：" << (t_01 - t_00) / getTickFrequency() << endl;

		// 自己实现的高斯
		int gaussKernelSize = 3;
		cv::Mat selfSmoothed = test_img.clone();

		auto gaussWeightX = std::make_unique<double[]>(gaussKernelSize);
		getKernal(gaussWeightX.get());

		// 在横向进行一次相加
		auto t_0 = getTickCount();

		const int byteLen = 1;// float 4 位
		for (int y = 0; y < test_img.rows; y++)
		{
			for (int x = 1; x < test_img.cols - 1; x++)
			{
				double newPix = 0.0;
				//// 边界处理后的对应的权值矩阵实际值
				newPix += (float)test_img.data[y*test_img.step[0] + (x - 1)* byteLen] * gaussWeightX[0];
				newPix += (float)test_img.data[y*test_img.step[0] + x * byteLen] * gaussWeightX[1];
				newPix += (float)test_img.data[y*test_img.step[0] + (x + 1)* byteLen] * gaussWeightX[2];
				selfSmoothed.data[y * selfSmoothed.step[0] + x * byteLen] = (uchar)newPix;
			}
		}
		// 在竖向进行一次相加
		for (int y = 1; y < test_img.rows - 1; y++)
		{
			for (int x = 0; x < test_img.cols; x++)
			{
				double newPix = 0.0f;
				//// 边界处理后的对应的权值矩阵实际值
				newPix += (float)selfSmoothed.data[(y - 1) * selfSmoothed.step[0] + x * byteLen] * gaussWeightX[0];
				newPix += (float)selfSmoothed.data[y * selfSmoothed.step[0] + x * byteLen] * gaussWeightX[1];
				newPix += (float)selfSmoothed.data[(y + 1) * selfSmoothed.step[0] + x * byteLen] * gaussWeightX[2];
				selfSmoothed.data[y * selfSmoothed.step[0] + x * byteLen] = (uchar)newPix;
			}
		}

		auto t_1 = getTickCount();
		cout << "手动计算高斯模糊耗时：" << (t_1 - t_0) / getTickFrequency() << endl;
		selfSmoothed.convertTo(selfSmoothed, CV_8UC1);

	}

}

int main()
{
	test_shape_match();
	//SL();
	//SL_2();
	// 
	//simulation_cuda_sobel();
	//simulationGuass();


	int thread_num = std::thread::hardware_concurrency();
	cout << "当前程序允许新开最高线程数：" << thread_num << endl;

	return 0;
}
