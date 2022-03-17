
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
	shapeInfoProducer(cv::Mat& src, int in_featuresNum, float magnitude, std::vector<float> _scale_range, std::vector<float> _angle_range, string inpath);
	~shapeInfoProducer(void);
	//shapeInfoProducer(cv::Mat& src, int in_featuresNum, float magnitude, float threshold) {};
	cv::Mat srcImg, pyramidImg;
	cv::Mat magnitudeImg;							//梯度幅值图
	cv::Mat quantized_angle;						// 量化后的角度图，根据离散量化投票机制，找出每个位置3x3领域中出现次数最多的方向(为了避免小的形变的影响)，如果投票数量小于5，方向则为0， [0-7]；
	cv::Mat angle_ori;								// 角度图, 
	float magnitude_value;							//选特征点时的幅值阈值
	int num_features;
	std::vector<float> scale_level;					// 模板的多比例
	int scale_level_num;								// 多比例的层数
	std::vector<float> angle_range;					// 模板的多比例
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
	void hysteresisGradient();

	bool extractFeaturePoints();

	bool selectScatteredFeatures(const std::vector<Candidate>& candidates, std::vector<Feature>& features,
		size_t num_features, float distance);


};


void shapeInfoProducer::hysteresisGradient()
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
			quant_r[c] &= 7;// 很巧妙地做了一个反转，如方向15的转为7,这里就是量化方向，360度转成16个方向，再变成8个方向
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	this->quantized_angle = Mat::zeros(this->angle_ori.size(), CV_8U);
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


	// 获取极大值
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
	// 过一遍非极大值抑制
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
	cv::cvtColor(this->pyramidImg, show_max, COLOR_GRAY2RGB);
	for (int i = 0; i < temp_candidates.size(); i++)
	{
		cv::Point show_coor = cv::Point(temp_candidates.at(i).f.x, temp_candidates.at(i).f.y);
		//show_max.at<Vec3b>(show_coor.y, show_coor.x) = cv::Scalar(0, 255, 0);
		show_max.at<Vec3b>(show_coor.y, show_coor.x)[0] = 0;
		show_max.at<Vec3b>(show_coor.y, show_coor.x)[1] = 255;
		show_max.at<Vec3b>(show_coor.y, show_coor.x)[2] = 255;
	}

	// We require a certain number of features
	if (temp_candidates.size() < num_features)
	{
		if (temp_candidates.size() <= 4) {
			std::cout << "too few features, abort" << std::endl;
			return false;
		}
		//std::cout << "have no enough features, exaustive mode" << std::endl;
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

	// 临时可视化最终找到的极大值特征
	cv::Mat show_two;
	cv::cvtColor(this->pyramidImg, show_two, COLOR_GRAY2RGB);
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


shapeInfoProducer::shapeInfoProducer(cv::Mat& in_src, int in_featuresNum, float in_magnitude, std::vector<float> _scale_range, std::vector<float> _angle_range, string inpath)
{
	this->srcImg = in_src;
	this->magnitude_value = in_magnitude;
	this->num_features = in_featuresNum;
	this->scale_level = _scale_range;
	this->scale_level_num = (int)_scale_range.size();
	this->angle_range = _angle_range;
	this->path = inpath;
	/*cout << "strat train, train img rows:" << this->srcImg.rows << endl;
	cout << "num_features:" << num_features << endl;
	cout << "magnitude_value:" << magnitude_value << endl;
	cout << "mul scale size:" << this->scale_level_num << endl;*/
	std::cout << "save feature path:" << this->path << endl;

}


shapeInfoProducer::~shapeInfoProducer(void)
{

	std::cout << "退出形状模板训练" << endl;
}


void shapeInfoProducer::train()
{
	// 保存训练的特征
	cv::FileStorage fs(this->path, cv::FileStorage::WRITE);
	fs << "pyramid_scales" << "[";
	fs << "[:";
	for (int i = 0; i < this->scale_level.size(); i++)
	{ 
		fs << this->scale_level[i];

	}
	fs << "]";
	fs << "]";
	fs << "template_pyramids" << "[";
	for (int pyramid_levels_num = 0; pyramid_levels_num < this->scale_level_num; pyramid_levels_num++)
	{
		float pyramid_levels_scale = this->scale_level[pyramid_levels_num];
		this->pyramidImg.release();

		cv::Mat smoothed;
		cv::resize(this->srcImg, this->pyramidImg, cv::Size(0, 0), pyramid_levels_scale, pyramid_levels_scale);

		//cv::Size size(this->srcImg.cols * pyramid_levels_scale, this->srcImg.rows * pyramid_levels_scale);
		//if ((int)this->pyramid_scale_level[pyramid_level] != 1)
		//cv::pyrDown(this->srcImg, this->pyramidImg, size);



		static const int KERNEL_SIZE = 5;
		cv::GaussianBlur(this->pyramidImg, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

		cv::Mat sobel_dx, sobel_dy;
		cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
		cv::phase(sobel_dx, sobel_dy, this->angle_ori, true);
		// 量化到8个方向，再根据领域找出现次数最多的方向，结果放在：quantized_angle
		hysteresisGradient();

		// todo 这里应加个循环的提取，金字塔和旋转的模板
		//  nms 根据幅值图筛选极大值特征点

		if (!this->extractFeaturePoints())
		{
			//return -1;
			std::cout << "训练失败" << endl;

		}
		else
		{
			fs << "{";
			fs << "template_id" << pyramid_levels_num; // 后面这里加循环写入模板增强
			fs << "pyramid_level" << pyramid_levels_num;
			fs << "template_width" << pyramidImg.cols;
			fs << "template_height" << pyramidImg.rows;

			fs << "features" << "[";
			for (int i = 0; i < this->out_features.size(); i++)
			{
				fs << "[:" << this->out_features[i].x << this->out_features[i].y <<
					this->out_features[i].label << "]";

			}
			fs << "]";
			fs << "}";
		}
		std::cout << "比例：" << pyramid_levels_scale << " 的模板训练完毕" << endl;

		// 释放多比例图片中的占用图，用于下一个金字塔的mat类型使用
		this->angle_ori.release();
		this->magnitudeImg.release();
		this->quantized_angle.release();
		this->out_features.clear();


	}
	fs << "]";

	fs.release();

}



class shapeMatch
{
public:
	shapeMatch(cv::Mat& in_testImg, float in_magnitude, float in_threshold, float in_iou, string in_feature_path);
	~shapeMatch(void);
	cv::Mat testImg;

	//cv::Mat magnitudeImg;			// 测试图梯度幅值图
	//cv::Mat angle_ori;				// 角度图，0-360度
	//cv::Mat quantized_angle;		// 测试图量化（0-360  ->  0-7）后的方向
	//cv::Mat spread_quantized;		// 测试图广播后的方向
	std::vector<float> pyramid_scale_level; // 模板的多比例
	int pyramid_scale_level_num;			// 多比例的层数
	float threshold;
	float magnitude;
	float iou;
	std::string feature_path;
	std::vector<std::pair<int, std::vector<std::vector<Feature>>>> in_features;		// 第一个float类型存的是金字塔的比例
	std::vector<cv::Size> template_size;
	std::vector<cv::Mat> vec_quantized_angle;			// 单个相应图
	std::vector<std::vector<cv::Mat>> vec_response_maps;		// 多比例的金字塔响应图，

	int _T = 1;
	void load_shapeInfos();

	// 处理测试图片的梯度方向，广播，响应图
	void inference();
	void quantizedGradientOrientations(cv::Mat& angle_ori, cv::Mat& quantized_angle, cv::Mat& magnitudeImg);
	void spread(const Mat& src, Mat& dst, int T);

	void orUnaligned8u(const uchar* src, const int src_stride, uchar* dst, const int dst_stride,
		const int width, const int height);

	void computeResponseMaps(const Mat& src, std::vector<Mat>& in_response_maps, int pyramid_level);

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


};


shapeMatch::shapeMatch(cv::Mat& in_testImg, float in_magnitude, float in_threshold, float in_iou, string in_feature_path)
{
	this->threshold = in_threshold;
	this->feature_path = in_feature_path;
	this->testImg = in_testImg;
	this->magnitude = in_magnitude;
	this->iou = in_iou;
	std::cout << "特征文件路径：" << this->feature_path << endl;
	std::cout << "输入的测试图梯度阈值：" << this->magnitude << endl;

	load_shapeInfos();
	this->vec_response_maps.resize(this->pyramid_scale_level_num);
	this->vec_quantized_angle.resize(this->pyramid_scale_level_num);
}

shapeMatch::~shapeMatch(void)
{
	std::cout << "退出形状模板匹配" << endl;
}


void shapeMatch::load_shapeInfos()
{
	cv::FileStorage fs(this->feature_path, cv::FileStorage::READ);
	FileNode fn = fs.root();
	FileNode tps_level = fn["pyramid_scales"];
	this->pyramid_scale_level.clear();
	{
		FileNodeIterator tps_level_it = tps_level.begin(), tps_level_it_end = tps_level.end();
	
		auto _feature_it = (*tps_level_it).begin();
		_feature_it >> this->pyramid_scale_level;
		
	}
	this->pyramid_scale_level_num = (int)pyramid_scale_level.size();

	FileNode tps_fn = fn["template_pyramids"];
	this->in_features.resize(tps_fn.size());
	this->template_size.resize(tps_fn.size());

	int expected_id = 0;
	vector<Feature> one_pyramid_features;
	FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
	for (; tps_it != tps_it_end; ++tps_it, ++expected_id)
	{
		int template_id = (*tps_it)["template_id"];
		int pyramid_level = (*tps_it)["pyramid_level"];
		CV_Assert(template_id == expected_id);
		int _template_width = (*tps_it)["template_width"];
		int _template_height = (*tps_it)["template_height"];
		this->template_size[pyramid_level].width = _template_width;
		this->template_size[pyramid_level].height = _template_height;
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
		this->in_features[pyramid_level].first = pyramid_level;
		this->in_features[pyramid_level].second.emplace_back(one_pyramid_features);
	}


}


void shapeMatch::quantizedGradientOrientations(cv::Mat& angle_ori, cv::Mat& quantized_angle, cv::Mat& magnitudeImg)//Mat &magnitude, Mat &quantized_angle, Mat &angle, float threshold
{
	// Quantize 360 degree range of orientations into 16 buckets
	// Note that [0, 11.25), [348.75, 360) both get mapped in the end to label 0,
	// for stability of horizontal and vertical features.
	Mat_<unsigned char> quantized_unfiltered;
	angle_ori.convertTo(quantized_unfiltered, CV_8U, 16 / 360.0);

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
	for (int r = 1; r < angle_ori.rows - 1; ++r)
	{
		uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
		for (int c = 1; c < angle_ori.cols - 1; ++c)
		{
			quant_r[c] &= 7;// 很巧妙地做了一个反转，如方向15的转为7,这里就是量化方向，360度换乘16块，再变成8块，也就是8个方向
		}
	}

	// Filter the raw quantized image. Only accept pixels where the magnitude is above some
	// threshold, and there is local agreement on the quantization.
	quantized_angle = Mat::zeros(angle_ori.size(), CV_8U);
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


	//int max_thread_num = (int)std::thread::hardware_concurrency() / 1;
	int max_thread_num = 1;
	int Rows = angle_ori.rows - 2;
	int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
	std::vector<std::thread> vec_threads;
	cv::Mat temp_magnitudeImg(magnitudeImg);
	cv::Mat temp_quantized_angle(quantized_angle);
	for (int thread_i = 0; thread_i < max_thread_num; thread_i++)
	{
		int startR = std::max(1, thread_i * per_thread_process_num);
		int endR = std::min((thread_i + 1) * per_thread_process_num, Rows);
		vec_threads.emplace_back([=, &temp_magnitudeImg, &quantized_unfiltered, &temp_quantized_angle]()
		{ subQuantiOri(startR, endR, angle_ori.cols, strong_magnitude_value, temp_magnitudeImg, quantized_unfiltered, temp_quantized_angle);
		});
	}
	for (auto& t : vec_threads)
	{
		t.join();
	}
	magnitudeImg = temp_magnitudeImg;
	quantized_angle = temp_quantized_angle;



}


void countSimilarity(std::vector<std::pair<int, std::vector<std::vector<Feature>>>>& in_features, std::vector<cv::Size>& template_size,
					cv::Size test_img_size, std::vector<cv::Mat>& response_maps, cv::Mat& similarity, int _T, int pyramidIdx)
{

	// 循环滑窗匹配,这是高层的滑窗匹配，金字塔Idx应该都是0；
	auto subThreadCountSimilarity = [=](int threadIdx, int startRow, int endRow, std::vector<std::pair<int, std::vector<std::vector<Feature>>>>& in_features, std::vector<cv::Size>& template_size,
										int test_img_cols, std::vector<cv::Mat>& response_maps, cv::Mat& similarity, int _T, int pyramidIdx)
	{
		for (int r = startRow; r < endRow; r += _T)
		{
		//cout << "threadIdx :" << threadIdx << ", " <<  endl;
			for (int c = 0; c < test_img_cols - template_size[pyramidIdx].width; c += _T)
			{
				for (int t = 0; t < in_features[pyramidIdx].second.size(); t++) // 这个循环是模板级别的
				{
					int fea_size = (int)in_features[pyramidIdx].second[t].size();
					int ori_sum = 0;
					for (int f = 0; f < fea_size; f++) // 这个循环是特征级别的
					{
						Feature feat = in_features[pyramidIdx].second[t][f];
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

	//int max_thread_num = (int)std::thread::hardware_concurrency() / 1;
	int max_thread_num = 1;
	int Rows = (int)test_img_size.height - template_size[pyramidIdx].height;
	int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
	//std::cout <<"当前电脑最大线程数："<< max_thread_num << " ,  每个线程最多处理行数：" <<per_thread_process_num <<", 总行数："<< Rows << endl;
	std::vector<std::thread> vec_threads;
	for (int thread_i = 0; thread_i < max_thread_num; thread_i++)
	{
		int startRow = thread_i * per_thread_process_num;
		int endRow = std::min((thread_i + 1) * per_thread_process_num, Rows);
		vec_threads.emplace_back([=, &in_features, &template_size, &response_maps, &similarity]()
		{ subThreadCountSimilarity(thread_i, startRow, endRow, in_features, template_size,
			test_img_size.width, response_maps, similarity, _T, pyramidIdx);
		});

	}

	for (auto& t : vec_threads)
	{
		t.join();
	}

}


void similarityLocal(cv::Point start_point, int count_field, std::vector<cv::Mat>& response_maps, cv::Mat& similarity, 
					 std::vector<std::pair<int, std::vector<std::vector<Feature>>>>& in_features, float high_scale)
{
	int count_mul = 2;
	for (int r = start_point.y; r < start_point.y + count_field * count_mul; r++)
	{

		for (int c = start_point.x; c < start_point.x + count_field * count_mul; c++)
		{
			for (int t = 0; t < in_features[1].second.size(); t++) // 这个循环是模板级别的
			{
				int fea_size = (int)in_features[1].second[t].size();
				int ori_sum = 0;
				for (int f = 0; f < fea_size; f++) // 这个循环是特征级别的
				{
					Feature feat = in_features[1].second[t][f];
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


}


void shapeMatch::inference()
{
	auto t_start = getTickCount();
	std::vector<cv::Size> vec_test_img_pyramid_size;
	vec_test_img_pyramid_size.clear();
	// 制作多个响应图
	for(int pyramid_level = 0; pyramid_level < this->pyramid_scale_level_num; pyramid_level ++ )
	{
		auto t0 = getTickCount();
		// 先下采样

		cv::Size size(this->testImg.cols * this->pyramid_scale_level[pyramid_level], this->testImg.rows * this->pyramid_scale_level[pyramid_level]);
		vec_test_img_pyramid_size.emplace_back(size);
		cv::Mat pyramidImg;
		if (this->pyramid_scale_level[pyramid_level] != 1)
			//cv::pyrDown(this->testImg, pyramidImg, size);
			cv::resize(this->testImg, pyramidImg, cv::Size(0, 0), this->pyramid_scale_level[pyramid_level], this->pyramid_scale_level[pyramid_level]);
		else
			pyramidImg = this->testImg.clone();
		cv::Mat smoothed;
		static const int KERNEL_SIZE = 5;
		cv::GaussianBlur(pyramidImg, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);

		cv::Mat sobel_dx, sobel_dy, magnitudeImg, angle_ori, quantized_angle, spread_quantized;
		auto t_sobel = cv::getTickCount();
		cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
		cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
		magnitudeImg = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);
		cv::phase(sobel_dx, sobel_dy, angle_ori, true);
		// 量化到8个方向，再根据领域找出现次数最多的方向，主要输出结果是：this->quantized_angle
		auto t_quatize = cv::getTickCount();
		std::cout << "求梯度耗时：" << (t_quatize - t_sobel) / cv::getTickFrequency() << endl;;
		quantizedGradientOrientations(angle_ori, quantized_angle, magnitudeImg);

		auto t_spread = cv::getTickCount();
		std::cout << "量化方向耗时：" << (t_spread - t_quatize) / cv::getTickFrequency() << endl;;
		spread(quantized_angle, spread_quantized, 2);		// 最后一个参数 是广播的领域尺度，2, 4 或 8

		auto t_response = cv::getTickCount();
		std::cout << "广播方向耗时：" << (t_response - t_spread) / cv::getTickFrequency() << endl;;
		computeResponseMaps(spread_quantized, this->vec_response_maps[pyramid_level], pyramid_level);
		auto t_response_out = cv::getTickCount();
		std::cout << "计算响应图耗时：" << (t_response_out - t_response) / cv::getTickFrequency() << endl;;

		auto t1 = getTickCount();
		auto t01 = (t1 - t0) / getTickFrequency();
		std::cout << "测试图方向量化、广播、8个方向响应图耗时：" << t01 << endl;
		// 到这一步后基本需要的都可以了，开始匹配，输入：训练的特征点，8个响应图

	}


	std::vector<matchResult> second_results;
	{
		// 1. 先在高层滑窗匹配匹配
		auto t_start_first_match = cv::getTickCount();
		cv::Mat first_similarity = cv::Mat::zeros(vec_test_img_pyramid_size[0], CV_32FC1);
		countSimilarity(this->in_features, this->template_size, vec_test_img_pyramid_size[0], this->vec_response_maps[0], first_similarity, _T, 0);

		auto t_first_count_similarity = cv::getTickCount();
		std::cout << "第一次计算相似度耗时：" << (t_first_count_similarity - t_start_first_match) / cv::getTickFrequency() << endl;

		// 2. 找极大值点 （这里直接在极大值点映射到低层去找，映射时会根据比例映射到一个范围；
		//					后续：在极大值点的3x3领域内做映射到底层的范围）
		std::vector<matchResult> first_results;
		{
			int _rows = first_similarity.rows;
			int _cols = first_similarity.cols;
			cv::Mat left = cv::Mat::zeros(first_similarity.size(), first_similarity.type());
			cv::Mat right = cv::Mat::zeros(first_similarity.size(), first_similarity.type());
			cv::Mat top = cv::Mat::zeros(first_similarity.size(), first_similarity.type());
			cv::Mat bottom = cv::Mat::zeros(first_similarity.size(), first_similarity.type());

			first_similarity.rowRange(0, _rows - 1).copyTo(top.rowRange(1, _rows));
			first_similarity.rowRange(1, _rows).copyTo(bottom.rowRange(0, _rows - 1));
			first_similarity.colRange(0, _cols - 1).copyTo(left.colRange(1, _cols));
			first_similarity.colRange(1, _cols).copyTo(right.colRange(0, _cols - 1));

			cv::Mat first_binary = first_similarity >= this->threshold
				& first_similarity >= left
				& first_similarity >= right
				& first_similarity >= top
				& first_similarity >= bottom;
			// 临时可视化看看找到的极大值点
			//cv::Mat element = getStructuringElement(MORPH_RECT, Size(15, 15)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
			//cv::Mat show_binary;
			//cv::dilate(first_binary, show_binary, element);

			// 3.采用连通域的方法，有时会有两个位置连续的，也就是置信度一样的，这样应该可以避免
			cv::Mat labels, status, centroids;
			int label_count = cv::connectedComponentsWithStats(first_binary, labels, status, centroids);
			for (int i = 1; i < label_count; ++i)
			{
				int _x = status.at<int>(i, CC_STAT_LEFT);
				int _y = status.at<int>(i, CC_STAT_TOP);
				first_results.emplace_back(_x, _y, first_similarity.at<float>(_y, _x));
			}
		}

		auto t_first_find_extreme = cv::getTickCount();
		std::cout << "第一次找相似度极大值耗时：" << (t_first_find_extreme - t_first_count_similarity) / cv::getTickFrequency() << endl;
		// 4. 根据高层极大值点，映射回到底层的范围，在底层的响应图做相似度计算；
		cv::Mat second_similarity = cv::Mat::zeros(vec_test_img_pyramid_size[1], CV_32FC1);

		float hight_map_to_low_scale = this->pyramid_scale_level[1] / this->pyramid_scale_level[0];
		int count_field = 15;	// 在高层金字塔的极大值点映射到底层点的5 * 5领域，计算金字塔底层的相似度

		for (int extreme_pointIdx = 0; extreme_pointIdx < first_results.size(); extreme_pointIdx++)
		{
			cv::Point low_extremePoint_left_up((int)(first_results[extreme_pointIdx].x * hight_map_to_low_scale - count_field),
											   (int)(first_results[extreme_pointIdx].y * hight_map_to_low_scale - count_field));

			similarityLocal(low_extremePoint_left_up, count_field, this->vec_response_maps[1], second_similarity,
				this->in_features, hight_map_to_low_scale);

		}

		auto t_second_count_similaty = cv::getTickCount();
		std::cout << "第二次计算相似度耗时：" << (t_second_count_similaty - t_first_find_extreme) / cv::getTickFrequency() << endl;
		cv::Mat second_similarity_extreme;
		second_similarity.convertTo(second_similarity_extreme, CV_8U, 200);
		// 5.再根据第二次的相似度图，找极大值， 需要优化第二次的极大值，图片大，耗时多
		{
			int _rows = second_similarity_extreme.rows;
			int _cols = second_similarity_extreme.cols;
			cv::Mat left = cv::Mat::zeros(second_similarity_extreme.size(), second_similarity_extreme.type());
			cv::Mat right = cv::Mat::zeros(second_similarity_extreme.size(), second_similarity_extreme.type());
			cv::Mat top = cv::Mat::zeros(second_similarity_extreme.size(), second_similarity_extreme.type());
			cv::Mat bottom = cv::Mat::zeros(second_similarity_extreme.size(), second_similarity_extreme.type());


			second_similarity_extreme.rowRange(0, _rows - 1).copyTo(top.rowRange(1, _rows));
			second_similarity_extreme.rowRange(1, _rows).copyTo(bottom.rowRange(0, _rows - 1));
			second_similarity_extreme.colRange(0, _cols - 1).copyTo(left.colRange(1, _cols));
			second_similarity_extreme.colRange(1, _cols).copyTo(right.colRange(0, _cols - 1));

			auto t_max_2 = cv::getTickCount();
			cv::Mat second_binary = second_similarity_extreme >= (uint8_t)(this->threshold*200)
				& second_similarity_extreme >= left
				& second_similarity_extreme >= right
				& second_similarity_extreme >= top
				& second_similarity_extreme >= bottom;
			auto t_max_3 = cv::getTickCount();
			std::cout << "第二次找极大值耗时：" << (t_max_3 - t_max_2) / cv::getTickFrequency() << endl;
			// 临时可视化看看找到的极大值点
			//cv::Mat element = getStructuringElement(MORPH_RECT, Size(15, 15)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
			//cv::Mat show_binary;
			//cv::dilate(second_binary, show_binary, element);

			// 统计极大值
			for (int r = 0; r < second_binary.rows; r++)
			{
				uchar* bin_r = second_binary.ptr<uchar>(r);
				for (int c = 0; c < second_binary.cols; c++)
				{
					if (bin_r[c] == 255)
					{
						second_results.emplace_back(c, r, second_similarity.at<float>(r, c));

					}
				}
			}

			auto t_max_4 = cv::getTickCount();
			std::cout << "找连通域耗时：" << (t_max_4 - t_max_3) / cv::getTickFrequency() << endl;
		}

	}

	// 6.根据底层的匹配结果，进行非极大值抑制，并根据IOU过滤重复的

	auto t_start_nms = getTickCount();
	cv::Mat show_img, iouShowImg;
	cv::cvtColor(this->testImg, show_img, COLOR_GRAY2RGB);
	
	std::sort(second_results.begin(), second_results.end(), [&](matchResult a, matchResult b) {return a.score > b.score; });
	auto _results = second_results;
	std::vector<int> del_results_idx ;
 	for (int n = 0; n < second_results.size()-1; n++)
	{
		cv::Rect box_0(second_results[n].x, second_results[n].y, this->template_size[0].width, this->template_size[0].height);
		for (int m = n + 1; m < second_results.size(); m++)
		{
			cv::Rect box_1(second_results[m].x, second_results[m].y, this->template_size[0].width, this->template_size[0].height);
			float twoArea = 2 * (float)this->template_size[0].width * (float)this->template_size[0].height;
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
		second_results.erase(second_results.begin() + (del_results_idx.at(d) - d));

	}

	auto t_end_nms = getTickCount();

	auto t_nms = (t_end_nms - t_start_nms) / getTickFrequency();
	std::cout << "NMS IOU 耗时：" << t_nms << endl;
	auto t_all = (t_end_nms - t_start) / getTickFrequency();

	std::cout << "形状匹配总耗时：" << t_all << endl;

	// 可视化匹配结果的点
	for (int i = 0; i < second_results.size(); i++)
	{
		int offset_x = second_results[i].x;
		int offset_y = second_results[i].y;
		int feat_size = (int)this->in_features[1].second[0].size();
		for (int f = 0; f < feat_size; f++)
		{
			Feature feat = this->in_features[1].second[0][f];
			int x = offset_x + feat.x;
			int y = offset_y + feat.y;
			//show_img.at<Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
			cv::circle(show_img, cv::Point(x, y), 2, cv::Scalar(0, 0, 255));
		}
		putText(show_img, std::to_string(second_results[i].score).substr(0, 6), cv::Point(second_results[i].x, second_results[i].y), 1, 5, cv::Scalar(0, 255, 0), 3);



	}
	auto t_show = (getTickCount()  - t_end_nms) / getTickFrequency();
	std::cout << "可视化 耗时：" << t_show << endl;
	std::cout << "总匹配数量：" << second_results.size() << endl;

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


void shapeMatch::computeResponseMaps(const Mat& src, std::vector<Mat>& in_response_maps, int pyramid_level)
{

	// Allocate response maps
	in_response_maps.resize(8);
	for (int i = 0; i < 8; ++i)
		in_response_maps[i].create(src.size(), CV_8U);

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

		//int max_thread_num = std::min(8, (int)std::thread::hardware_concurrency() / 1);
		int max_thread_num = 1;
		int Rows = 8;
		int per_thread_process_num = (int)std::ceil((float)Rows / (float)max_thread_num);
		std::vector<std::thread> vec_threads;
		auto temp_response_maps = this->vec_response_maps[pyramid_level];
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
		this->vec_response_maps[pyramid_level] = temp_response_maps;



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

	//int max_thread_num = (int)std::thread::hardware_concurrency() / 1;
	int max_thread_num = 1;
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
	//string mode = "test";  
	 string mode = "train";

	if (mode == "train")
	{
		//cv::Mat template_img = cv::imread("F:\\1heils\\sheng_shape_match\\ganfa/下半部分.png", 0);//规定输入的是灰度图，三通道的先不弄
		cv::Mat template_img = cv::imread("F:\\1heils\\sheng_shape_match\\ganfa/上半部分.png", 0);//规定输入的是灰度图，三通道的先不弄
		shapeInfoProducer trainer(template_img, 64, 30, {0.3, 1}, "F:\\1heils\\sheng_shape_match\\ganfa/上半部分.yaml");
		trainer.train();
	}


	else if (mode == "test")
	{
		cv::Mat test_img = cv::imread("F:\\1heils\\sheng_shape_match\\ganfa/test_3.png", 0);	// sl_template_test     sl_test_4
		shapeMatch tester(test_img, 30, 0.9f, 0.1f, "F:\\1heils\\sheng_shape_match\\ganfa\\上半部分.yaml");
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
