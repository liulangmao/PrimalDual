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
	printf("PrimalDualAl in.png [method] out.png\n");
	printf("Application{0,1}:0 for PrimalDual_ROF ;\n 1 for  OpticalFlow_\n");
}


int main(int argc, char *argv[]) {
	usage();
	if (argc < 1) { // Check the value of argc. If not enough parameters have been passed, inform user and exit.
		printf("I didn't get enough parameters.");
		return 0;
	}
	else { // if we got enough parameters...
		string InputFileName, OutputFileName,InputFileName2;
		InputFileName = string(argv[1]);
		Application algo;
        if (argc == 4)
		{
			OutputFileName = string(argv[3]);
			algo = (Application)atoi(argv[2]);
		}
		else if(argc == 5)
		{
			OutputFileName = string(argv[4]);
			InputFileName2 = string(argv[2]);
			algo = (Application)atoi(argv[3]);
		}
#ifdef Debuge
		printf("InputFileName:%s.", InputFileName.c_str());
		printf("OutputFileName:%s.", OutputFileName.c_str());
		printf("InputFileName2:%s.", InputFileName2.c_str());
#endif
		switch (algo){
		case Application::PrimalDual_ROF :{
				cv::Mat result = Add_GaussiaNoise(InputFileName);
				PrimalDualAl *pridual = new PrimalDualAl(result, 100, 0.05, 5, Method::HuberRof);
				//pridual->Denoise();
			}
		case Application::OpticalFlow_:{
				cv::Mat I1 = ReadColor2Gray(InputFileName);
				cv::Mat I2 = ReadColor2Gray(InputFileName2);
				OpticalFlow *optical = new OpticalFlow(I1, I2,Method::TV_L1);
				optical->ComputeTv_l1_motion_primal_dual();
			}
		}
			
	}

	return 0;
}