#include "Header.h"

typedef Eigen::SparseMatrix<float> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<float> T;
class PrimalDualAl
{
public:
	PrimalDualAl(cv::Mat & inputImg, int steps, float afa, float lamda, Method method);
	~PrimalDualAl();
	void Denoise();
private:
	void SetParas();
	void PreCompute();
	void IterConvergence(Eigen::MatrixXf & convergeValue);
	void testX_Y(int rtest, int ctest);
	float FirstValue();
	float DualValue();
private: 
	cv::Mat m_InputImg;
	cv::Mat m_OutputImg;
	int m_NumStep;
	float m_delta;
	float m_gamma;
	Method m_method;
	float m_L;
	float m_mu;
	float m_tau;
	float m_sigma;
	float m_theta;
	int m_width;
	int m_height;
	Eigen::MatrixXf m_XdimenData;
	SpMat m_YFirstAndSeconderivateData;
	Eigen::MatrixXf m_YIterCoef_line;//dual variables
	Eigen::MatrixXf m_result;
	Eigen::MatrixXf m_result_line;//primal variables
	SpMat m_divop;
};

