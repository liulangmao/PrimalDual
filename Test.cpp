#include "PrimalDualAl.h"
#include "OpticalFlow.h"
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/opencv.hpp>
#include <string>
using namespace std;

cv::Mat Add_GaussiaNoise(string InputFileName)
{
	cv::Mat InputImg = cv::imread(InputFileName, CV_LOAD_IMAGE_COLOR);
	InputImg.convertTo(InputImg, CV_32FC3);
	cv::cvtColor(InputImg, InputImg, CV_BGR2GRAY);
	InputImg = InputImg / 255.0;
	cv::Mat noise = cv::Mat(InputImg.size(), CV_64F);
    cv:: Mat result;
	normalize(InputImg, result, 0.0, 1.0, CV_MINMAX, CV_64F);
	cv::randn(noise, 0, 0.1);
	result = result + noise;
	normalize(result, result, 0.0, 1.0, CV_MINMAX, CV_64F);
	
#if Debuge==1
	cv::Mat outprint;
	result.convertTo(outprint, CV_32F, 255, 0);
	cv::imwrite("result.png", outprint);
#else
	result.convertTo(result, CV_32F, 1, 0);
#endif
	return result;
}

cv::Mat ReadColor2Gray(string InputFileName)
{
	cv::Mat InputImg = cv::imread(InputFileName, CV_LOAD_IMAGE_COLOR);
	InputImg.convertTo(InputImg, CV_32FC3);
	cv::cvtColor(InputImg, InputImg, CV_BGR2GRAY);
	InputImg = InputImg / 255.0;
	return InputImg;
}
void usage() {
	printf("-Alogrithm 0  -Input1 in.png  -Output Denoising.png\n");
	printf("Or  -Alogrithm 1  -Input1 frame10.png -Input2 frame11.png -Output OpticalFlow.png \n");
	printf("Application{0,1}:0 for PrimalDual_ROF for denoise;\n 1 for  OpticalFlow_\n");
}
void str_split(const string& input, const string& delimiters,
	vector<string>& results)
{
	string::size_type pos;
	size_t size = input.size();
	for (size_t i = 0; i<size; ++i)
	{
		pos = input.find(delimiters, i); //从第i个位置查找delimiters分割符第一次出现的位置；  
		if (pos<size)
		{
			string s = input.substr(i, pos - i);//把从i开始，长度为pos-i的元素拷贝给s;  
			results.push_back(s);
			i = pos;
		}
	}

}
map<string, string> parse_switches(int argc, const char **argv) {
	map<string, string> ans;
	for (int i = 1; i < argc; i++) {
		if (strlen(argv[i]) && argv[i][0] == '-' && i + 1 < argc) {
			ans[argv[i]] = argv[i + 1];
			i += 1;
		}
	}
	printf("switches size: %d\n", ans.size());
	return ans;
}
void set_from_switches(string s,map<string, string>& switches)
{
	vector<string> L;
	string delimiters = " ";
	str_split(s, delimiters, L);
	vector<const char *> argv;
	for (int i = 0; i < (int)L.size(); i++) {
		if (L[i].size() && !(L[i].size() == 1 && L[i][0] == ' ')) {
			argv.push_back(L[i].c_str());
		}
	}
	switches = parse_switches(argv.size(), &argv[0]);
}

int main(int argc, char *argv[]) {
	usage();
	if (argc < 1) { // Check the value of argc. If not enough parameters have been passed, inform user and exit.
		printf("I didn't get enough parameters.");
		return 0;
	}
	else { // if we got enough parameters...
		string InputFileName, OutputFileName,InputFileName2;

		Application algo;
		map<string, string> switches;
		switches = parse_switches(argc, (const char **)argv);
		//set_from_switches(argv, switches);
		if (switches.count("-Alogrithm"))  { algo = (Application)atoi(switches["-Alogrithm"].c_str()); }
		if (switches.count("-Output")) { OutputFileName = switches["-Output"].c_str(); }
		if (switches.count("-Input1")) { InputFileName = switches["-Input1"].c_str(); }
		if (switches.count("-Input2")) { InputFileName2 = switches["-Input2"].c_str(); }
       
#ifdef Debuge
		printf("InputFileName:%s.", InputFileName.c_str());
		printf("OutputFileName:%s.", OutputFileName.c_str());
		printf("InputFileName2:%s.", InputFileName2.c_str());
#endif
		switch (algo){
		case Application::PrimalDual_ROF :{
				cv::Mat result = Add_GaussiaNoise(InputFileName);
				PrimalDualAl *pridual = new PrimalDualAl(result, 100, 0.05, 5, Method::HuberRof);
				pridual->Denoise();
				break;
			}
		case Application::OpticalFlow_:{
				cv::Mat I1 = ReadColor2Gray(InputFileName);
				cv::Mat I2 = ReadColor2Gray(InputFileName2);
				OpticalFlow *optical = new OpticalFlow(I1, I2,Method::TV_L1);
				int level = 1000;
				optical->CoarseToFine(level, 0.9);
			//	optical->ComputeTv_l1_motion_primal_dual();
			break;
			}
		default: printf("wrong input");

		}
			
	}

	return 0;
}