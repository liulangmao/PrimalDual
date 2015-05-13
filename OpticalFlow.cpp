#include "OpticalFlow.h"
#define IterationMax 5 

OpticalFlow::OpticalFlow(cv::Mat & I1, cv::Mat &I2, Method m_method)
{
	m_Height = I1.rows;
	m_Width = I1.cols;
	if (I1.channels() == 3)
	{
		cv::cvtColor(I1, Illum1, CV_BGR2Lab);
		cv::cvtColor(I2, Illum2, CV_BGR2Lab);
	}
	else if (I1.channels()==1)
	{
		Illum1 = I1/255.0;
		Illum2 = I2/255.0;
		
	}
	else
	{
		printf("I don't know the input's format!");
	}
	m_N = m_Width*m_Height;



	m_lamda = 0.15;
	m_tau = 0.25;
	m_theta = 0.3;
}

void OpticalFlow::MakeNabla()
{
	std::vector<T> coefficients;
	m_nabla.resize(m_N * 2, m_N);
	int count = 0;
	for (int r = 0; r <= m_Height - 1; r++)
	{
		for (int c = 0; c <= m_Width - 1; c++)
		{
			count = c + r*m_Width;
			if (r < m_Height - 1)
			{
				if (count >= 2 * m_N - 1 || count + m_Width >= m_N)
				{
					printf("boundary error!");
				}
				coefficients.push_back(T(count, count, -1));
				coefficients.push_back(T(count, count + m_Width, 1));
			}
			if (c < m_Width - 1)
			{
				if (count + m_Width*m_Height >= 2 * m_N - 1 || count + 1 >= m_N)
				{
					printf("boundary error!");
				}
				coefficients.push_back(T(count + m_Width*m_Height, count, -1));
				coefficients.push_back(T(count + m_Width*m_Height, count + 1, 1));
			}
		}
	}
	m_nabla.setFromTriplets(coefficients.begin(), coefficients.end());
}
void OpticalFlow::ComputeTv_l1_motion_primal_dual()
{
	m_iteration = IterationMax;

#ifdef Debuge
	Eigen::MatrixXf Engery = Eigen::MatrixXf::Zero(m_iteration, 1);
#endif
	m_gradientFlowUX = Eigen::MatrixXf::Zero(m_N, 1);
	m_gradientFlowUY = Eigen::MatrixXf::Zero(m_N, 1);

	//m_valueP = Eigen::MatrixXf::Zero(2 * m_N, m_N);
	m_valueU = Eigen::MatrixXf::Zero(2 * m_N, 1);
	
	Eigen::MatrixXf m_gradientFlowUX_Nplus1 = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_gradientFlowUY_Nplus1 = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_gradientFlowUPX_Nplus1 = Eigen::MatrixXf::Zero(2 * m_N, 1);
	Eigen::MatrixXf m_gradientFlowUPY_Nplus1 = Eigen::MatrixXf::Zero(2 * m_N, 1);
	//cv::Mat valueUMat(2 * m_N, m_N, CV_32FC1, cv::Scalar::all(0));

	Eigen::MatrixXf valueU0 = Eigen::MatrixXf::Zero(2 * m_N, 1);
	float L_square = 8;
	m_sigma = 1 / (L_square*m_tau);

	MakeNabla();//Compute K or K_star here!
	Eigen::MatrixXf WrapNextframe2 = Eigen::MatrixXf::Zero(m_N, 1);
	Mat2OneRowMatrixf(Illum2, WrapNextframe2);
	Wrapping(m_gradientFlowUX, m_gradientFlowUY, m_gradientFlowUX_Nplus1, WrapNextframe2, m_gradientFlowUY_Nplus1, m_valueU);

	

	valueU0 = m_valueU;
	//
	//
	Eigen::MatrixXf pho = Eigen::MatrixXf::Zero(m_N, 1);
#ifdef Debuge
	pho = (m_valueU.block(0, 0, m_N, 1) - m_gradientFlowUX).cwiseProduct(m_gradientFlowUX_Nplus1) + (m_valueU.block(m_N, 0, m_N, 1) - m_gradientFlowUY).cwiseProduct(m_gradientFlowUY_Nplus1);
	Eigen::MatrixXf y1 = m_gradientFlowUX_Nplus1.transpose()*m_gradientFlowUX_Nplus1;
	Eigen::MatrixXf y2 = m_gradientFlowUY_Nplus1.transpose()*m_gradientFlowUY_Nplus1;
	float temp = y1(0, 0) + y2(0, 0);
	
	float firstpart = std::sqrt(temp);
	y1 = pho.transpose()*pho;
	float secondpart = m_lamda*y1(0,0);
	Engery(0, 0) = firstpart + secondpart;
	printf("iteration %d: %.3f\n The first part engery is %.3f\n, The second part engery is %.3f\n ", m_iteration, Engery(0, 0), firstpart, secondpart);
	Show(m_gradientFlowUX_Nplus1, m_gradientFlowUY_Nplus1, 0);
#endif	
	//
	// Steps on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.
	Eigen::MatrixXf m_valueP_temp = Eigen::MatrixXf::Zero(2*m_N, 1);;

	for (int iter = 0; iter< m_iteration; iter++)
	{
		//First to update p
		m_gradientFlowUPX_Nplus1 = m_gradientFlowUPX_Nplus1 + m_sigma*m_nabla*valueU0.block(0,0,m_N,1);
		m_gradientFlowUPY_Nplus1 = m_gradientFlowUPY_Nplus1 + m_sigma*m_nabla*valueU0.block(m_N, 0, m_N, 1);
		// Steps1 on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.
		for (int n = 0; n < m_N; n++)
		{
			// By equation on page23 in paper<A first-order primal-dual algorithm for convex problems> to compute the discrete version of the isotropic total variation norm.
			m_valueP_temp(n, 0) = std::sqrt(m_gradientFlowUPX_Nplus1(n, 0)*m_gradientFlowUPX_Nplus1(n, 0) + m_gradientFlowUPY_Nplus1(n , 0)*m_gradientFlowUPY_Nplus1(n , 0));
			if (m_valueP_temp(n, 0) <= 1)
			{
				m_valueP_temp(n, 0) = 1;
			}
			m_valueP_temp(n + m_N, 0) = m_valueP_temp(n, 0);
			m_gradientFlowUPX_Nplus1(n, 0) = m_gradientFlowUPX_Nplus1(n, 0) / m_valueP_temp(n, 0);
			m_gradientFlowUPY_Nplus1(n, 0) = m_gradientFlowUPY_Nplus1(n + m_N, 0) / m_valueP_temp(n + m_N, 0);
		}
		//Second to update U
		valueU0 = m_valueU;
		m_divop = m_nabla.transpose();
		// Part steps2 on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.
		valueU0.block(0, 0, m_N, 1) = m_valueU.block(0, 0, m_N, 1) - m_tau*m_divop*m_gradientFlowUPX_Nplus1;
		valueU0.block(m_N, 0, m_N, 1) = m_valueU.block(m_N, 0, m_N, 1) - m_tau*m_divop*m_gradientFlowUPY_Nplus1;
		//Steps3 Another part has been changed following the equation 11 on page5  Or equation 16 on page 6 in paper<An Improved Algorithm for TV-L1 Optical Flow>.
		Eigen::MatrixXf pho = Eigen::MatrixXf::Zero(m_N, 1);
		std::cout << "total Number is:" << m_N << std::endl;
		pho = (m_valueU.block(0, 0, m_N, 1) - m_gradientFlowUX).cwiseProduct(m_gradientFlowUX_Nplus1) + (m_valueU.block(m_N, 0, m_N, 1) - m_gradientFlowUY).cwiseProduct(m_gradientFlowUY_Nplus1);
		int case1 = 0, case2 = 0, case3 = 0,case4=0;
		for (int n = 0; n < m_N; n++)
		{
			float I1_Nabla = std::max(m_gradientFlowUX_Nplus1(n, 0)*m_gradientFlowUX_Nplus1(n, 0) + m_gradientFlowUY_Nplus1(n, 0)*m_gradientFlowUY_Nplus1(n, 0), (float)1e-09);

			if (pho(n, 0) < -m_lamda*m_theta*I1_Nabla)
			{
				m_valueU(n, 0) = m_valueU(n, 0) + m_lamda*m_theta*m_gradientFlowUX_Nplus1(n,0);
				m_valueU(n + m_N) = m_valueU(n + m_N) + m_lamda*m_theta*m_gradientFlowUY_Nplus1(n,0);
				case1++;
			}
			else if (pho(n, 0) > m_lamda*m_theta*I1_Nabla)
			{
				m_valueU(n, 0) = m_valueU(n, 0) - m_lamda*m_theta*m_gradientFlowUX_Nplus1(n, 0);
				m_valueU(n + m_N) = m_valueU(n + m_N) - m_lamda*m_theta*m_gradientFlowUY_Nplus1(n, 0);
				case2++;
			}
			else 
			{
				m_valueU(n, 0) = m_valueU(n, 0) - pho(n, 0)*m_gradientFlowUX_Nplus1(n, 0) / I1_Nabla;
				m_valueU(n + m_N) = m_valueU(n + m_N) - m_lamda*m_theta*m_gradientFlowUY_Nplus1(n, 0) / I1_Nabla;
				case3++;
			}
			if (pho(n, 0) == 0 )
			{
				//std::cout << pho(n, 0) << std::endl;
				case4++;
			}
		}
		std::cout << "Case1 is:" << case1 << std::endl;
		std::cout << "Case2 is:" << case2 << std::endl;
		std::cout << "Case3 is:" << case3 << std::endl;
		std::cout << "Case4 is:" << case4 << std::endl;
		//Steps 4 Update U for next iteration
		valueU0 = m_valueU + m_theta*(m_valueU - valueU0);

		//
#ifdef Debuge

		Eigen::MatrixXf y1 = m_gradientFlowUX_Nplus1.transpose()*m_gradientFlowUX_Nplus1;
		Eigen::MatrixXf y2 = m_gradientFlowUY_Nplus1.transpose()*m_gradientFlowUY_Nplus1;
		float temp = y1(0, 0) + y2(0, 0);

		float firstpart = std::sqrt(temp);

		y1 = pho.transpose()*pho;
		float y = y1(0, 0);
		float secondpart = m_lamda*y1(0, 0);
		Engery(iter, 0) = firstpart + secondpart;

		printf("iteration %d: %.3f\n The first part engery is %.3f\n, The second part engery is %.3f\n y is%.3f ", iter, Engery(iter, 0), firstpart, secondpart,y);
		Show(m_gradientFlowUX_Nplus1, m_gradientFlowUY_Nplus1, iter);

		
#endif

		m_gradientFlowUX = m_gradientFlowUX_Nplus1;
		m_gradientFlowUY = m_gradientFlowUY_Nplus1;
		Wrapping(m_gradientFlowUX, m_gradientFlowUY, m_gradientFlowUX_Nplus1, WrapNextframe2, m_gradientFlowUY_Nplus1, m_valueU);
		
	}

    
}
void OpticalFlow::Show(Eigen::MatrixXf& m_gradientFlowUX_Nplus1, Eigen::MatrixXf& m_gradientFlowUY_Nplus1, int m_iteration)
{
	cv::Mat MatY(m_gradientFlowUY_Nplus1.rows(), m_gradientFlowUY_Nplus1.cols(), CV_32FC1, m_gradientFlowUY_Nplus1.data());
	cv::Mat MatX(m_gradientFlowUX_Nplus1.rows(), m_gradientFlowUX_Nplus1.cols(), CV_32FC1, m_gradientFlowUX_Nplus1.data());
	//calculate angle and magnitude
	cv::Mat magnitude, angle;
	cv::cartToPolar(MatX, MatY, magnitude, angle, true);

	//translate magnitude to range [0;1]
	double min, max; cv::Point p_min, p_max;
	cv::minMaxLoc(magnitude, &min, &max, &p_min, &p_max);
	printf(" max :%.3f, min:%.3f\n", max, min);
	magnitude.convertTo(magnitude, -1, 1.0 / max);

	//build hsv image
	cv::Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
	_hsv[2] = magnitude;
	cv::merge(_hsv, 3, hsv);

	//convert to BGR and show
	cv::Mat bgr;//CV_32FC3 matrix
	cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
	char file_name[50];
	sprintf(file_name, "ConvergePic%d.png", m_iteration);
	//m_OutputImg.convertTo(m_OutputImg, CV_32F, 255, 0);
	cv::imwrite(file_name, bgr*255.0);
	//cv::imshow("optical flow", bgr);
}
bool OpticalFlow::Isboundary(int &i, int &j)
/*i= width or cols
  j= height or rows*/
{
	bool is_x_out = false,is_y_out=false;
	if (i >= m_Width-1)
	{
		is_x_out = true;
		i = m_Width - 1;
	}
	else if (i <= 0)
	{
		i = 0;
		is_x_out = true;
	}
	else 
	{
		is_x_out = false;
	}


	if (j >= m_Height - 1)
	{
		is_y_out = true;
		j = m_Height - 1;
	}
	else if (j <= 0)
	{
		j = 0;
		is_y_out = true;
	}
	else
	{
		is_y_out = false;
	}
	if (is_x_out || is_y_out)
		return true;
	
}
void OpticalFlow::Mat2OneRowMatrixf(cv::Mat Matrix, Eigen::MatrixXf& MatrixEigen)
{
	int width = Matrix.cols;
	int height = Matrix.rows;
	int index = 0;
	for (int h = 0; h < height; h++)
	{
		for (int w = 0; w < width; w++)
		{
			index = w + h*width;
			MatrixEigen(index, 0) = Matrix.at<float>(h, w);
		}
	}
}
void OpticalFlow::Wrapping(Eigen::MatrixXf& m_gradientFlowUX, Eigen::MatrixXf& m_gradientFlowUY, Eigen::MatrixXf& I_x, Eigen::MatrixXf& I_2wrap, Eigen::MatrixXf& I_y, Eigen::MatrixXf &valueUMat)
{
	/*m_gradientFlowUX and m_gradientFlowUY are the optical_flow.
 I_x is another input which is next frame.
 I_2wrap is the wrap result after ordinate corresponding.
 I_x and I_y are the two output.
 */
	cv::Mat dst,dst1,dst2;
	cv::Mat map_x_original, map_y_original,map_x_addone,map_x_subone, map_y_addone,map_y_subone;
	Eigen::MatrixXf WrappedImgFromGradient, WrappedImgFromGradientXADD, WrappedImgFromGradientXSUB, WrappedImgFromGradientYADD, WrappedImgFromGradientYSUB;

	dst.create(Illum2.size(), Illum2.type());
	map_x_original.create(Illum2.size(), CV_32FC1);
	map_y_original.create(Illum2.size(), CV_32FC1);

	map_x_addone.create(Illum2.size(), CV_32FC1);
	map_y_addone.create(Illum2.size(), CV_32FC1);

	map_x_subone.create(Illum2.size(), CV_32FC1);
	map_y_subone.create(Illum2.size(), CV_32FC1);

	WrappedImgFromGradient = Eigen::MatrixXf::Zero(m_N,1);
	WrappedImgFromGradientXADD = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientXSUB = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientYADD = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientYSUB = Eigen::MatrixXf::Zero(m_N, 1);
	/// Update m_gradientFlowUX & m_gradientFlowUY. Then apply remap
	int index = 0;
	for (int r = 0; r < dst.rows; r++)
	{
		for (int c = 0; c < dst.cols; c++)
		{
			index = c + r*m_Width;
			int row = m_gradientFlowUY(index, 0) + r;
			int col = m_gradientFlowUX(index, 0) + c;
			bool is_boundary = Isboundary(col, row);
			map_x_original.at<float>(r, c) = col;
			map_y_original.at<float>(r, c) = row;
			row = row+1;
		    col = col+1;
			is_boundary = Isboundary(col, row);
			map_x_addone.at<float>(r, c) = col;
			map_y_addone.at<float>(r, c) = row;
			row = row - 2;
			col = col - 2;
		    is_boundary = Isboundary(col, row);
			map_x_subone.at<float>(r, c) = col;
			map_y_subone.at<float>(r, c) = row;

		}
	}
	remap(Illum2, dst, map_x_original, map_y_original, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, I_2wrap);

	// By equation 14 in paper<TV-L1 Optical Flow Estimation> to compute the gradient of the Image I1.
	remap(Illum2, dst1, map_x_addone, map_y_original, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst1, WrappedImgFromGradientXADD);
	remap(Illum2, dst, map_x_subone, map_y_original, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, WrappedImgFromGradientXSUB);
	I_x = WrappedImgFromGradientXADD*0.5 - WrappedImgFromGradientXSUB*0.5;

	Eigen::MatrixXf yy = Eigen::Map<Eigen::MatrixXf>(I_x.data(), m_N, 1);
	valueUMat.block(0, 0, m_N, 1) = I_x.block(0, 0, m_N,1);
	//valueUMat.block(0, 0, m_N,0) = I_x;




	remap(Illum2, dst1, map_x_original, map_y_addone, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst1, WrappedImgFromGradientYADD);
	remap(Illum2, dst, map_x_original, map_y_subone, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, WrappedImgFromGradientYSUB);
	I_y = WrappedImgFromGradientYADD*0.5 - WrappedImgFromGradientYSUB*0.5;
	cv::Mat mat2 = dst1*0.5 - dst*0.5;
	valueUMat.block(m_N, 0, m_N, 1) = I_y.block(0, 0, m_N, 1);



	index = 0;
	for (int r = 0; r < dst.rows; r++)
	{
		for (int c = 0; c < dst.cols; c++)
		{
			index = c + r*m_Width;
			if (Isboundary(c, r))
			{
				I_2wrap(index, 0) = 0;
				I_x(index, 0) = 0;
				I_y(index, 0) = 0;
			}
		}
	}

	
}
OpticalFlow::~OpticalFlow()
{
}
