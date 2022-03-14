
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
	static const int KERNEL_SIZE = 5;
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
	shapeMatch(cv::Mat& in_testImg, float in_magnitude, float in_threshold, float in_iou, string in_feature_path);
	~shapeMatch(void);
	cv::Mat testImg;

	cv::Mat magnitudeImg;			// 测试图梯度幅值图
	cv::Mat angle_ori;				// 角度图，0-360度
	cv::Mat quantized_angle;		// 测试图量化（0-360  ->  0-7）后的方向
	cv::Mat spread_quantized;		// 测试图广播后的方向
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


};


shapeMatch::shapeMatch(cv::Mat& in_testImg, float in_magnitude, float in_threshold, float in_iou, string in_feature_path)
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

shapeMatch::~shapeMatch(void)
{
	cout << "退出形状模板匹配" << endl;
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


	auto subQuantiOri = [=](int startR, int endR, int cols, float magnitude_value, cv::Mat& _magnitudeImg, cv::Mat& _quantized_unfiltered, cv::Mat& _quantized_angle) 
	{

		for (int r = startR; r < endR; ++r)
		{
			float* mag_r = _magnitudeImg.ptr<float>(r);

			for (int c = 1; c < cols - 1; ++c)
			{
				if (mag_r[c] > magnitude_value)
				{
					// Compute histogram of quantized bins in 3x3 patch around pixel
					int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
					uchar* patch3x3_row = _quantized_unfiltered.ptr() + _quantized_unfiltered.step1() * (r - 1) + c - 1; // 太巧妙了
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += _quantized_unfiltered.step1();
					histogram[patch3x3_row[0]]++;
					histogram[patch3x3_row[1]]++;
					histogram[patch3x3_row[2]]++;

					patch3x3_row += _quantized_unfiltered.step1();
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
						_quantized_angle.at<uchar>(r, c) = uchar(1 << index);
				}
			}
		}
	};


	int max_thread_num = (int)std::thread::hardware_concurrency() / 1;
	int Rows = this->angle_ori.rows - 2;
	int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
	std::vector<std::thread> vec_threads;
	cv::Mat temp_magnitudeImg(this->magnitudeImg);
	cv::Mat temp_quantized_angle(this->quantized_angle);
	for (int thread_i = 0; thread_i < max_thread_num; thread_i++)
	{
		int startR = std::max(1, thread_i * per_thread_process_num);
		int endR = std::min((thread_i + 1) * per_thread_process_num, Rows);
		vec_threads.emplace_back([=, &temp_magnitudeImg, &quantized_unfiltered, &temp_quantized_angle]()
		{ subQuantiOri(startR, endR, this->angle_ori.cols, strong_magnitude_value, temp_magnitudeImg, quantized_unfiltered, temp_quantized_angle);
		});
	}
	for (auto& t : vec_threads)
	{
		t.join();
	}
	this->magnitudeImg = temp_magnitudeImg;
	this->quantized_angle = temp_quantized_angle;



}


void countSimilarity(std::vector<std::vector<std::vector<Feature>>>& in_features, std::vector<std::pair<int, int>>& template_size, cv::Mat& quantized_angle,
						std::vector<cv::Mat>& response_maps, cv::Mat& similarity, int _T, int pyramidIdx)
{

	// 循环滑窗匹配
	auto subThreadCountSimilarity = [=](int threadIdx, int startRow, int endRow, std::vector<std::vector<std::vector<Feature>>>& in_features, std::vector<std::pair<int, int>>& template_size, 
										cv::Mat& quantized_angle, std::vector<cv::Mat>& response_maps, cv::Mat& similarity, int _T, int pyramidIdx)
	{
		for (int r = startRow; r < endRow; r += _T)
		{
		//cout << "threadIdx :" << threadIdx << ", " <<  endl;
			for (int c = 0; c < quantized_angle.cols - template_size[pyramidIdx].first; c += _T)
			{
				for (int t = 0; t < in_features[pyramidIdx].size(); t++) // 这个循环是模板级别的
				{
					int fea_size = (int)in_features[pyramidIdx][t].size();
					int ori_sum = 0;
					for (int f = 0; f < fea_size; f++) // 这个循环是特征级别的
					{
						Feature feat = in_features[pyramidIdx][t][f];
						int label = feat.label;
						auto _ori = (int)response_maps[label].ptr<uchar>(r + feat.y)[c + feat.x];
						ori_sum += (int)response_maps[label].ptr<uchar>(r + feat.y)[c + feat.x];

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
	
	};

	int max_thread_num = (int)std::thread::hardware_concurrency() / 1;
	int Rows = (int)quantized_angle.rows - template_size[pyramidIdx].second;
	int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
	cout <<"当前电脑最大线程数："<< max_thread_num << " ,  每个线程最多处理行数：" <<per_thread_process_num <<", 总行数："<< Rows << endl;
	std::vector<std::thread> vec_threads;
	for (int thread_i = 0; thread_i < max_thread_num; thread_i++)
	{
		int startRow = thread_i * per_thread_process_num;
		int endRow = std::min((thread_i + 1) * per_thread_process_num, Rows);
		vec_threads.emplace_back([=, &in_features, &template_size, &quantized_angle, &response_maps, &similarity]()
		{ subThreadCountSimilarity(thread_i, startRow, endRow, in_features, template_size,
			quantized_angle, response_maps, similarity, _T, pyramidIdx);
		});

	}

	for (auto& t : vec_threads)
	{
		t.join();
	}

}


void shapeMatch::inference()
{
	auto t0 = getTickCount();
	Mat smoothed;
	static const int KERNEL_SIZE = 5;
	cv::GaussianBlur(this->testImg, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

	cv::Mat sobel_dx, sobel_dy;
	auto t_sobel = cv::getTickCount();
	cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
	cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
	this->magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
	cv::phase(sobel_dx, sobel_dy, this->angle_ori, true);
	// 量化到8个方向，再根据领域找出现次数最多的方向，主要输出结果是：this->quantized_angle
	auto t_quatize = cv::getTickCount();
	cout << "求梯度耗时：" << (t_quatize - t_sobel) / cv::getTickFrequency() << endl;;
	quantizedGradientOrientations();

	auto t_spread = cv::getTickCount();
	cout << "量化方向耗时：" << (t_spread - t_quatize) / cv::getTickFrequency() << endl;;
	spread(this->quantized_angle, this->spread_quantized, 1);		// 4 是广播的领域尺度，4 或 8

	auto t_response = cv::getTickCount();
	cout << "广播方向耗时：" << (t_response - t_spread) / cv::getTickFrequency() << endl;;
	computeResponseMaps(this->spread_quantized, this->response_maps);
	auto t_response_out = cv::getTickCount();
	cout << "计算响应图耗时：" << (t_response_out - t_response) / cv::getTickFrequency() << endl;;

	auto t1 = getTickCount();
	auto t01 = (t1 - t0) / getTickFrequency();
	cout << "测试图方向量化、广播、8个方向响应图耗时：" << t01 << endl;
	// 到这一步后基本需要的都可以了，开始匹配，输入：训练的特征点，8个响应图

	cv::Mat similarity = cv::Mat::zeros(this->testImg.size(), CV_32FC1);
	

	for (int p = 0; p < in_features.size(); p++) //这个循环是金字塔级别的
	{
		countSimilarity(this->in_features, this->template_size, this->quantized_angle, this->response_maps, similarity, _T, p);

	}



	auto t_count_similarity = cv::getTickCount();
	cout << "计算相似度耗时：" << (t_count_similarity - t_response_out) / cv::getTickFrequency() << endl;
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
	cv::Mat show_img, iouShowImg;
	cvtColor(this->testImg, show_img, COLOR_GRAY2RGB);
	
	std::sort(results.begin(), results.end(), [&](matchResult a, matchResult b) {return a.score > b.score; });
	auto _results = results;
	std::vector<int> del_results_idx ;
 	for (int n = 0; n < results.size()-1; n++)
	{
		cv::Rect box_0(results[n].x, results[n].y, this->template_size[0].first, this->template_size[0].second);
		for (int m = n + 1; m < results.size(); m++)
		{
			cv::Rect box_1(results[m].x, results[m].y, this->template_size[0].first, this->template_size[0].second);
			float twoArea = 2 * (float)this->template_size[0].first * (float)this->template_size[0].second;
			int iouWidth = std::min(box_1.x + box_1.width, box_0.x + box_0.width) - std::max(box_0.x, box_1.x);
			int iouHeight = std::min(box_1.y + box_1.height, box_0.y + box_0.height) - std::max(box_0.y, box_1.y);
			if (iouWidth < 0 || iouHeight < 0)
				continue;
			float _iou = (iouWidth * iouHeight) / (twoArea- (iouWidth * iouHeight));

			if (_iou >= this->iou)
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
			//show_img.at<Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
			cv::circle(show_img, cv::Point(x, y), 2, cv::Scalar(0, 0, 255));
		}
		putText(show_img, std::to_string(results[i].score).substr(0, 6), cv::Point(results[i].x, results[i].y), 1, 5, cv::Scalar(0, 255, 0), 3);

		//cout << "匹配到第" << i << "个目标的置信度: " << std::to_string(results[i].score).substr(0, 6) << endl;

	}
	cout << "总匹配数量：" << results.size() << endl;

	float newWindowWidth, newWindowHeight, fixed = 1000;
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
	cv::resizeWindow("show_img", cv::Size((int)newWindowWidth, (int)newWindowHeight));
	cv::imshow("show_img", show_img);
	cv::waitKey(0);

	cv::imwrite("F:\\1heils\\sheng_shape_match\\ganfa\\save.png", show_img);

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

	cv::Mat lsb4(src.size(), CV_8U);// 低位的四个方向，00001111
	cv::Mat msb4(src.size(), CV_8U);// 高位的四个方向，11110000

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

	//auto t_lsb_start = cv::getTickCount();
	{
		uchar* lsb4_data = lsb4.ptr<uchar>();
		uchar* msb4_data = msb4.ptr<uchar>();

		// LUT is designed for 128 bits SIMD, so quite triky for others
		// For each of the 8 quantized orientations...
		auto subFindMaxOri = [=](int startOriMap, int endOriMap, int rows, int cols, uchar* _lsb4_data, uchar* _msb4_data, std::vector<cv::Mat>& _response_maps)
		{
			for (int ori = startOriMap; ori < endOriMap; ++ori) 
			{
				uchar* map_data = _response_maps[ori].ptr<uchar>();
				const uchar* lut_low = SIMILARITY_LUT + 32 * ori;
				for (int i = 0; i < rows * cols; ++i)
				{
					// 查表，论文里面是通过不同方向求cos值，但这里不一样，用一个表（8个方向，每个方向有32种结果？），
					// 求最大响应来表示测试图的方向广播图在不同方向下的响应；结果已经预算好放到表中，直接读结果就行
					//
					// 广播后的一个像素方向根据8bit前后分两份，然后每一份有16种可能的方向，一个像素的前后两个方向 查找 表中 方向对应的响应，
					// 然后求两个方向最大的响应，得出该像素在8个方向中某个方向的响应值；
					//  
					map_data[i] = std::max(lut_low[_lsb4_data[i]], lut_low[_msb4_data[i] + 16]);

				}
			}
		};

		int max_thread_num = std::min(8, (int)std::thread::hardware_concurrency() / 1);
		int Rows = 8;
		int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
		std::vector<std::thread> vec_threads;
		auto temp_response_maps = this->response_maps;
		for (int thread_i = 0; thread_i < max_thread_num; thread_i++)
		{
			int startR = thread_i * per_thread_process_num;
			int endR = std::min((thread_i + 1) * per_thread_process_num, Rows);
			vec_threads.emplace_back([=, &temp_response_maps]()
			{ 
				subFindMaxOri(startR, endR, src.rows, src.cols, lsb4_data, msb4_data, temp_response_maps);
			});
		}
		for (auto& t : vec_threads)
		{
			t.join();
		}
		this->response_maps = temp_response_maps;



	}
	//auto t_lsb_end = cv::getTickCount();
	//cout << "计算前后四个方向" << (t_lsb_end - t_lsb_start) / cv::getTickFrequency() << endl;

}



void shapeMatch::spread(const Mat& src, Mat& dst, int T)
{
	// T 代表广播方向的领域，如2X2，4x4,论文是8x8,实际使用中8太大了，会导致识别偏移；
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
	auto subOrSpread = [=](int startR, int endR, const uchar* src, const int src_stride, uchar* dst,
							const int dst_stride, const int width)
	{
		src += startR * src_stride;
		dst += startR * dst_stride;
		for (int r = startR; r < endR; ++r)
		{
			for (int c = 0; c < width; c++)
				dst[c] |= src[c];

			// Advance to next row
			src += src_stride;
			dst += dst_stride;
		}
	};

	int max_thread_num = (int)std::thread::hardware_concurrency() / 1;
	int Rows = height;
	int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
	std::vector<std::thread> vec_threads;
	for (int thread_i = 0; thread_i < max_thread_num; thread_i++)
	{
		int startR = thread_i * per_thread_process_num;
		int endR = std::min((thread_i + 1) * per_thread_process_num, Rows);
		vec_threads.emplace_back([=, &src, &dst]()
		{ subOrSpread(startR, endR, src, src_stride, dst, dst_stride, width);
		});
	}
	for (auto& t : vec_threads)
	{
		t.join();
	}

}


void test_shape_match()
{
	string mode = "test";  
	 //string mode = "train";

	if (mode == "train")
	{
		cv::Mat template_img = cv::imread("F:\\1heils\\sheng_shape_match\\ganfa/下半部分.png", 0);//规定输入的是灰度图，三通道的先不弄
		shapeInfoProducer trainer(template_img, 64, 30, "F:\\1heils\\sheng_shape_match\\ganfa/下半部分.yaml");
		trainer.train();

	}


	else if (mode == "test")
	{
		cv::Mat test_img = cv::imread("F:\\1heils\\sheng_shape_match\\ganfa/test_3.png", 0);	// sl_template_test     sl_test_4
		shapeMatch tester(test_img, 30, 0.9f, 0.1, "F:\\1heils\\sheng_shape_match\\ganfa/下半部分.yaml");
		tester.inference();

	}


}



int main()
{
	test_shape_match();


 // 	int thread_num = std::thread::hardware_concurrency();
	//cout << "当前程序允许新开最高线程数：" << thread_num << endl;
	
	return 0;
}
