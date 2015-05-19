#pragma once
#include "Header.h"
#define UNKNOWN_FLOW_THRESH 1e9

//#define MAXCOLS 60

class OpticalFlow
{
public:
	OpticalFlow(cv::Mat & I1, cv::Mat &I2, Method m_method);
	~OpticalFlow();
	void MakeNabla();
	void Wrapping(Eigen::MatrixXf& m_gradientFlowUX, Eigen::MatrixXf& m_gradientFlowUY, Eigen::MatrixXf& I_x, Eigen::MatrixXf& I_2wrap, Eigen::MatrixXf& I_y, Eigen::MatrixXf& I_Basic);
	void ComputeTv_l1_motion_primal_dual();
	bool Isboundary(int& i, int &j);
	void Mat2OneRowMatrixf(cv::Mat Matrix, Eigen::MatrixXf& MatrixEigen);
	cv::Mat OneColMatrix2Mat(Eigen::MatrixXf& m_gradientFlowUX_Nplus1,int width, int col);
	void ShowFlow(Eigen::MatrixXf& m_gradientFlowUX_Nplus1, Eigen::MatrixXf& m_gradientFlowUY_Nplus1, int m_iteration, int wrap);
	void DebugePrintMatrix(int Width, Eigen::MatrixXf & Matri);
	bool unknown_flow(float u, float v) {
		return (fabs(u) >  UNKNOWN_FLOW_THRESH)
			|| (fabs(v) >  UNKNOWN_FLOW_THRESH)
			|| isnan(u) || isnan(v);
	}
	void computeColor(float fx, float fy, uchar *pix);
	void makecolorwheel();
	void setcols(int r, int g, int b, int k)
	{
		colorwheel[k][0] = r;
		colorwheel[k][1] = g;
		colorwheel[k][2] = b;
	}
private:
	int m_Width; 
	int m_Height;
	int m_N;
	Method m_method;
	cv::Mat Illum1;
	cv::Mat Illum2;

	int m_NumStep;
	float m_lamda;
	float m_theta;
	float m_tau;
	float m_sigma;
	float m_criteria;
	int m_iteration;
	Eigen::MatrixXf m_gradientFlowUX;
	Eigen::MatrixXf m_gradientFlowUY;
	//Eigen::MatrixXf m_opticalFlow;
	Eigen::MatrixXf m_valueP;
	SpMat m_nabla;//it is [2*N,N] which is really independent.
	SpMat m_divop;
	int ncols = 0;
	int colorwheel[60][3];
	//float m_delta;
	//float m_gamma;
	//float m_L;
	//float m_mu;
	
	//float m_sigma;
};

