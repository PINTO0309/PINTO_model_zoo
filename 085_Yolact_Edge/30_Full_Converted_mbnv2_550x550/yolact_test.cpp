#include <gflags/gflags.h>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

using namespace InferenceEngine;

const int num_class = 81;
const int num_priors = 19248;
const float score_threshold = 0.3f;
const int topk = 100;

const float aspect_ratios[3] = { 1.f, 0.5f, 2.f };
const float scales[5] = { 24.f, 48.f, 96.f, 192.f, 384.f };
const int conv_w_list[5] = { 69, 35, 18, 9, 5 };
const int conv_h_list[5] = { 69, 35, 18, 9, 5 };
const float center_variance = 0.1f;
const float size_variance = 0.2f;

struct Object
{
	int label;
	float prob;
	cv::Mat mask;
	cv::Rect_<float> rect;
	std::vector<float> maskdata;

};
static const char* class_names[] = { "background",
	"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant",
	"stop sign", "parking meter", "bench", "bird", "cat", "dog",
	"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat",
	"baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
	"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
	"hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush" };

static const unsigned char colors[19][3] = {
	{244,  67,  54},
	{233,  30,  99},
	{156,  39, 176},
	{103,  58, 183},
	{ 63,  81, 181},
	{ 33, 150, 243},
	{  3, 169, 244},
	{  0, 188, 212},
	{  0, 150, 136},
	{ 76, 175,  80},
	{139, 195,  74},
	{205, 220,  57},
	{255, 235,  59},
	{255, 193,   7},
	{255, 152,   0},
	{255,  87,  34},
	{121,  85,  72},
	{158, 158, 158},
	{ 96, 125, 139}
};
int draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	cv::Mat image = bgr.clone();

	int color_index = 0;

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		if (obj.prob < 0.2)
			continue;

		const unsigned char* color = colors[color_index++];

		cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y),
			cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

		// draw mask
		for (int y = 0; y < image.rows; y++)
		{
			const uchar* mp = obj.mask.ptr<uchar>(y);
			cv::Vec3b* p = image.ptr<cv::Vec3b>(y);
			for (int x = 0; x < image.cols; x++)
			{
				if (mp[x] == 255)
				{
					p[x] = cv::Vec3b(p[x][0] * 0.5 + color[0] * 0.5, p[x][1] * 0.5 + color[1] * 0.5, p[x][2] * 0.5 + color[2] * 0.5);
				}
			}
		}
	}

	//cv::imwrite("result.png", image);
	cv::imshow("image", image);
	int key = cv::waitKey(1);
	return key;
}


cv::Mat makePriors(int num_priors, int height,int width)
{
	cv::Mat priorbox(num_priors, 4, CV_32FC1);
	float* pPriorbox = (float*)priorbox.data;
	for (int n = 0; n < 5; n++)
	{
		int conv_w = conv_w_list[n];
		int conv_h = conv_h_list[n];
		float scale = scales[n];

		for (int i = 0; i < conv_h; i++)
		{
			for (int j = 0; j < conv_w; j++)
			{
				float x = (j + 0.5f) / conv_w;
				float y = (i + 0.5f) / conv_h;

				for (int k = 0; k < 3; k++)
				{
					float ar = aspect_ratios[k];
					ar = std::sqrt(ar);

					float w = scale * ar / width;
					float h = scale / ar / height;
					h = w;
					pPriorbox[0] = x;
					pPriorbox[1] = y;
					pPriorbox[2] = w;
					pPriorbox[3] = h;
					pPriorbox += 4;
				}
			}
		}
	}
	return priorbox;
}
static inline float intersection_area(const Object& a, const Object& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
	picked.clear();

	const int n = objects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = objects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = objects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = objects[picked[j]];

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			//             float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}
void decode(const float* confidence, const float* location,const float* priorBox,const float* mask,int width,int height,std::vector<Object> &objects)
{
	std::vector< std::vector<Object> > candidateBox;
	candidateBox.resize(num_class);
	for (int i = 0; i < num_priors; i++)
	{
		const float* conf = confidence + i* num_class;
		const float* box = location + i*4;
		const float* priors = priorBox + i*4;
		const float* maskdata = mask + i*32;

		int label = 0;
		float score = 0.f;
		for (int j = 1; j < num_class; j++)
		{
			float class_score = conf[j];
			if (class_score > score)
			{
				label = j;
				score = class_score;
			}
		}

		if (label == 0 || score <= score_threshold)
			continue;

		float x_center = center_variance * box[0] * priors[2] + priors[0];
		float y_center = center_variance * box[1] * priors[3] + priors[1];
		float w = (float)(std::exp(size_variance * box[2]) * priors[2]);
		float h = (float)(std::exp(size_variance * box[3]) * priors[3]);

		float xmin = x_center - w * 0.5f;
		float ymin = y_center - h * 0.5f;
		float xmax = x_center + w * 0.5f;
		float ymax = y_center + h * 0.5f;


		xmin = std::max(std::min(xmin * width, (float)(width - 1)), 0.f);
		ymin = std::max(std::min(ymin * height, (float)(height - 1)), 0.f);
		xmax = std::max(std::min(xmax * width, (float)(width - 1)), 0.f);
		ymax = std::max(std::min(ymax * height, (float)(height - 1)), 0.f);

		Object obj;
		obj.label = label;
		obj.prob = score;
		obj.rect.x = xmin;
		obj.rect.y = ymin;
		obj.rect.width = (xmax - xmin);
		obj.rect.height = (ymax - ymin);
		obj.maskdata = std::vector<float>(maskdata, maskdata + 32);

		candidateBox[label].push_back(obj);
	}

	for (int i = 0; i < candidateBox.size(); i++)
	{
		std::vector<Object>& candidates = candidateBox[i];
		std::sort(candidates.begin(), candidates.end(), [](const Object & a, const Object & b) { return a.prob > b.prob; });

		std::vector<int> picked;
		nms_sorted_bboxes(candidates, picked, 0.4f);

		for (int j = 0; j < (int)picked.size(); j++)
		{
			int z = picked[j];
			objects.push_back(candidates[z]);
		}
	}
	std::sort(objects.begin(), objects.end(), [](const Object & a, const Object & b) { return a.prob > b.prob; });

	if (topk < objects.size())
	{
		objects.resize(topk);
	}
}

void main()
{


	Core ie;
	InferRequest::Ptr inferRequest;

	CNNNetwork network = ie.ReadNetwork(".\\yolact_edge_550.xml");

	ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
	std::string inName = inputShapes.begin()->first;
	SizeVector & inSizeVector = inputShapes.begin()->second;
	inSizeVector[0] = 1;
	network.reshape(inputShapes);

	InputInfo & inputInfo = *network.getInputsInfo().begin()->second;
	inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
	inputInfo.setLayout(Layout::NHWC);
	inputInfo.setPrecision(Precision::U8);
	OutputsDataMap outputsDataMap = network.getOutputsInfo();
	for (auto& output : outputsDataMap) {
		output.second->setPrecision(Precision::FP32);
	}
	ExecutableNetwork executableNetwork = ie.LoadNetwork(network, "CPU");
	inferRequest = executableNetwork.CreateInferRequestPtr();

	cv::Mat priorBox = makePriors(num_priors, inSizeVector[2], inSizeVector[3]);

	cv::Mat img;
	cv::VideoCapture cap(0);
	while (1)
	{
		cap >> img;
		if (img.empty())
			continue;

		inferRequest->SetBlob(inName, wrapMat2Blob(img));
		inferRequest->Infer();

		Blob::Ptr locationBlob = inferRequest->GetBlob("828");     //4*19248
		Blob::Ptr maskmapsBlob = inferRequest->GetBlob("Conv_265");//138*138*32
		Blob::Ptr maskBlob = inferRequest->GetBlob("830");         //32*19248
		Blob::Ptr confidenceBlob = inferRequest->GetBlob("832");   //81*19248
		std::vector<Object> objects;
		decode(as<MemoryBlob>(confidenceBlob)->rmap().as<float*>(),as<MemoryBlob>(locationBlob)->rmap().as<float*>(),
			(float*)priorBox.data,as<MemoryBlob>(maskBlob)->rmap().as<float*>(),img.cols, img.rows, objects);
		const float* pMaskmaps = as<MemoryBlob>(maskmapsBlob)->rmap().as<float*>();

		for (int i = 0; i < objects.size(); i++)
		{
			Object& obj = objects[i];

			cv::Mat mask(138, 138, CV_32FC1);
			{
				mask = cv::Scalar(0.f);

				for (int p = 0; p < 32; p++)
				{
					const float* maskmap = pMaskmaps + p * 138 * 138;
					float coeff = obj.maskdata[p];
					float* mp = (float*)mask.data;
					for (int j = 0; j < 138 * 138; j++)
					{
						mp[j] += maskmap[j] * coeff;
					}
				}
			}

			cv::Mat mask2;
			cv::resize(mask, mask2, cv::Size(img.cols, img.rows));
			obj.mask = cv::Mat(img.rows, img.cols, CV_8UC1);
			{
				obj.mask = cv::Scalar(0);

				for (int y = 0; y < img.rows; y++)
				{
					if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
						continue;

					const float* mp2 = mask2.ptr<const float>(y);
					uchar * bmp = obj.mask.ptr<uchar>(y);

					for (int x = 0; x < img.cols; x++)
					{
						if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
							continue;

						bmp[x] = mp2[x] > 0.5f ? 255 : 0;
					}
				}
			}
		}
		int key = draw_objects(img, objects);
		if (key == 27)break;
	}

}