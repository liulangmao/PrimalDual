#include "OpticalFlow.h"
#define IterationMax 5 
#define WrapIteration 1
OpticalFlow::OpticalFlow(cv::Mat & I1, cv::Mat &I2, Method m_method)
{
	m_Height = I1.rows;
	m_Width = I1.cols;
	/*m_Height = 10;
	m_Width = 20;*/
	if (I1.channels() == 3)
	{
		cv::cvtColor(I1, Illum1, CV_BGR2Lab);
		cv::cvtColor(I2, Illum2, CV_BGR2Lab);
	}
	else if (I1.channels()==1)
	{
		Illum1.create(m_Height, m_Width, CV_32FC1);
		Illum2.create(m_Height, m_Width, CV_32FC1);
		//printf("\n");
		for (int r = 0; r < m_Height; r++)
		{
			for (int c = 0; c < m_Width; c++)
			{
				Illum1.at<float>(r, c) = I1.at<float>(r, c);
				//printf("%f ",Illum1.at<float>(r, c));
				Illum2.at<float>(r, c) = I2.at<float>(r, c);
			}
			//printf("\n");
		}
		//Illum1 = I1;
		//Illum2 = I2;
	}
	else
	{
		printf("I don't know the input's format!");
	}
	m_N = m_Width*m_Height;



	m_lamda = 1;
	m_tau = 1/std::sqrt(8);
	m_theta = 1 / std::sqrt(8);
}

void OpticalFlow::MakeNabla()
{
	/*The order is very important!!*/
	std::vector<T> coefficients;
	m_nabla.resize(m_N * 2, m_N);
	int count = 0;
	for (int r = 0; r <= m_Height - 1; r++)
	{
		for (int c = 0; c <= m_Width - 1; c++)
		{
			count = c + r*m_Width;
			if (c < m_Width - 1)
			{
				/*if (count >= 2 * m_N - 1 || count + 1 >= m_N)
				{
					printf("boundary error!");
				}*/
				coefficients.push_back(T(count, count, -1));
				coefficients.push_back(T(count, count + 1, 1));
			}
			if (r < m_Height - 1)
			{
				/*if (count + m_Width*m_Height >= 2 * m_N - 1 || count + m_Width >= m_N)
				{
					printf("boundary error!");
				}*/
				coefficients.push_back(T(count + m_Width*m_Height, count, -1));
				coefficients.push_back(T(count + m_Width*m_Height, count + m_Width, 1));
			}
		}
	}
	m_nabla.setFromTriplets(coefficients.begin(), coefficients.end());
}
void OpticalFlow::DebugePrintMatrix(int Width, Eigen::MatrixXf & Matri)
{
	printf("\n ");
	for (int r = 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf(" ... ");
		for (int c = 17; c < 20; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf("\n");
	}
	for (int r = 7; r < 10; r++)
	{
		for (int c = 0; c < 3; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf(" ... ");
		for (int c = 17; c < 20; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf("\n");
	}
}
void OpticalFlow::ComputeTv_l1_motion_primal_dual()
{
	m_iteration = IterationMax;

#ifdef Debuge
	Eigen::MatrixXf Engery = Eigen::MatrixXf::Zero(m_iteration, 1);
#endif
	m_gradientFlowUX = Eigen::MatrixXf::Zero(m_N, 1);
	m_gradientFlowUY = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_gradientFlowUX0 = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_gradientFlowUY0 = Eigen::MatrixXf::Zero(m_N, 1);
	//m_valueP = Eigen::MatrixXf::Zero(2 * m_N, m_N);
	
	Eigen::MatrixXf m_WrapX = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_WrapY = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_Wrap_Basic = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_gradientFlowUPX_Nplus1 = Eigen::MatrixXf::Zero(2 * m_N, 1);
	Eigen::MatrixXf m_gradientFlowUPY_Nplus1 = Eigen::MatrixXf::Zero(2 * m_N, 1);
	
	float L_square = 8;
	m_sigma = 1 / std::sqrt(L_square);

	MakeNabla();//Compute K or K_star here!
	Eigen::MatrixXf WrapNextframe2 = Eigen::MatrixXf::Zero(m_N, 1);
	for (int i = 0; i < WrapIteration; i++)
	{
		m_gradientFlowUX0 = m_gradientFlowUX;
		m_gradientFlowUY0 = m_gradientFlowUY;
		Wrapping(m_gradientFlowUX0, m_gradientFlowUY0, m_WrapX, WrapNextframe2, m_WrapY, m_Wrap_Basic);

		Eigen::MatrixXf pho = Eigen::MatrixXf::Zero(m_N, 1);

		// Steps on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.
		Eigen::MatrixXf m_valueX_temp = Eigen::MatrixXf::Zero(m_N, 1);
		Eigen::MatrixXf m_valueY_temp = Eigen::MatrixXf::Zero(m_N, 1);
		Eigen::MatrixXf m_valuePX = Eigen::MatrixXf::Zero(2 * m_N, 1);
		Eigen::MatrixXf m_valuePY = Eigen::MatrixXf::Zero(2 * m_N, 1);
		for (int iter = 0; iter< m_iteration; iter++)
		{
			//First to update p
			m_valuePX = m_valuePX + m_sigma*m_nabla*m_valueX_temp;
			m_valuePY = m_valuePY + m_sigma*m_nabla*m_valueY_temp;
			// Steps1 on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.
			for (int n = 0; n < m_N; n++)
			{
				// By equation on page23 in paper<A first-order primal-dual algorithm for convex problems> to compute the discrete version of the isotropic total variation norm.
				float temp = 0.0;
				temp = std::sqrt(m_valuePX(n, 0)*m_valuePX(n, 0) + m_valuePX(n + m_N, 0)*m_valuePX(n + m_N, 0) + m_valuePY(n, 0)*m_valuePY(n, 0) + m_valuePY(n + m_N, 0)*m_valuePY(n + m_N, 0));
				if (temp <= 1.0)
				{
					temp = 1.0;
				}
				m_valuePX(n, 0) = m_valuePX(n, 0) / temp;
				m_valuePX(n + m_N, 0) = m_valuePX(n + m_N, 0) / temp;
				m_valuePY(n, 0) = m_valuePY(n + m_N, 0) / temp;
				m_valuePY(n + m_N, 0) = m_valuePY(n + m_N, 0) / temp;
			}

			//Second to update U (which is what we want) using P
			// Part steps2 on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.
			m_divop = m_nabla.transpose();
			m_gradientFlowUX = m_gradientFlowUX - m_tau*m_divop*m_valuePX;
			m_gradientFlowUY = m_gradientFlowUY - m_tau*m_divop*m_valuePY;
			//Steps3 Another part has been changed following the equation 11 on page5  Or equation 16 on page 6 in paper<An Improved Algorithm for TV-L1 Optical Flow>.
			//Construct pho and update the U using pho
			Eigen::MatrixXf pho = Eigen::MatrixXf::Zero(m_N, 1);
			pho = (m_gradientFlowUX - m_gradientFlowUX0).cwiseProduct(m_WrapX) + (m_gradientFlowUY - m_gradientFlowUY0).cwiseProduct(m_WrapY) + WrapNextframe2;
			int case1 = 0, case2 = 0, case3 = 0, case4 = 0;
			for (int n = 0; n < m_N; n++)
			{
				float I1_Nabla = std::max(m_WrapX(n, 0)*m_WrapX(n, 0) + m_WrapY(n, 0)*m_WrapY(n, 0), (float)1e-09);

				if (pho(n, 0) < -m_lamda*m_theta*I1_Nabla)
				{
					m_gradientFlowUX(n, 0) = m_gradientFlowUX(n, 0) + m_lamda*m_theta*m_WrapX(n, 0);
					m_gradientFlowUY(n, 0) = m_gradientFlowUY(n, 0) + m_lamda*m_theta*m_WrapY(n, 0);
					case1++;
				}
				else if (pho(n, 0) > m_lamda*m_theta*I1_Nabla)
				{
					m_gradientFlowUX(n, 0) = m_gradientFlowUX(n, 0) - m_lamda*m_theta*m_WrapX(n, 0);
					m_gradientFlowUY(n, 0) = m_gradientFlowUY(n, 0) - m_lamda*m_theta*m_WrapY(n, 0);
					case2++;
				}
				else
				{
					m_gradientFlowUX(n, 0) = m_gradientFlowUX(n, 0) - pho(n, 0)*m_WrapX(n, 0) / I1_Nabla;
					m_gradientFlowUY(n, 0) = m_gradientFlowUY(n, 0) - pho(n, 0)*m_WrapY(n, 0) / I1_Nabla;
					case3++;
				}
				if (pho(n, 0) == 0)
				{
					case4++;
				}
			}
			//Steps 4 Update tempU for next iteration
			m_valueX_temp = 2 * m_gradientFlowUX - m_valueX_temp;
			m_valueY_temp = 2 * m_gradientFlowUY - m_valueY_temp;
			//
#ifdef Debuge
			std::cout << "Case1 is:" << case1 << std::endl;
			std::cout << "Case2 is:" << case2 << std::endl;
			std::cout << "Case3 is:" << case3 << std::endl;
			std::cout << "Case4 is:" << case4 << std::endl;
			//printf("I_x:");
			//DebugePrintMatrix(m_Width, m_gradientFlowUX);
			/*printf("I_y:");
			DebugePrintMatrix(m_Width, m_gradientFlowUY);*/

			printf("iteration %d:\n", iter);
			Eigen::MatrixXf y1 = m_gradientFlowUX.transpose()*m_gradientFlowUX;
			Eigen::MatrixXf y2 = m_gradientFlowUY.transpose()*m_gradientFlowUY;
			printf("The x gradient is %.3f \n", y1(0, 0));
			printf("The y gradient is %.3f \n", y2(0, 0));

			float temp = y1(0, 0) + y2(0, 0);

			float firstpart = std::sqrt(temp);
			printf("The first part engery is %.3f\n", firstpart);

			y1 = pho.transpose()*pho;
			float y = y1(0, 0);
			float secondpart = m_lamda*y1(0, 0);
			Engery(iter, 0) = firstpart + secondpart;

			printf("The second part engery is %.3f\n ", y);
			printf("The engery is %.3f\n ", Engery(iter, 0));

			ShowFlow(m_gradientFlowUX, m_gradientFlowUY, iter, i);
			Eigen::MatrixXf m_WrapX = Eigen::MatrixXf::Zero(m_N, 1);
			m_WrapX = (m_gradientFlowUX - m_gradientFlowUX0).cwiseProduct(m_WrapX);
			Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(1, m_Width);
			Eigen::MatrixXf a = Eigen::MatrixXf::Map(m_WrapX.data(), m_Height, m_Width, stride);
			m_WrapX = (m_gradientFlowUY - m_gradientFlowUY0).cwiseProduct(m_WrapY);
			Eigen::MatrixXf b = Eigen::MatrixXf::Map(m_WrapX.data(), m_Height, m_Width, stride);
			Eigen::MatrixXf c = Eigen::MatrixXf::Map(m_Wrap_Basic.data(), m_Height, m_Width, stride);

			cv::Mat wrapBasic = cv::Mat::zeros(m_Height,m_Width,CV_32FC1);
			cv::Mat wrapx = cv::Mat::zeros(m_Height, m_Width, CV_32FC1);
			cv::Mat wrapy = cv::Mat::zeros(m_Height, m_Width, CV_32FC1);
			cv::eigen2cv(a, wrapx);
			cv::eigen2cv(b, wrapy);
			cv::eigen2cv(c, wrapBasic);
			wrapBasic = wrapBasic + wrapx + wrapy;
			char file_name[50];
			sprintf(file_name, "Wrap%d_%d_%.3f.png", i, iter, Engery(iter, 0));
			cv::imwrite(file_name, wrapBasic*255.0);
#endif
	}
	
	}

    
}
void OpticalFlow::ShowFlow(Eigen::MatrixXf& m_gradientFlowUX_Nplus1, Eigen::MatrixXf& m_gradientFlowUY_Nplus1, int m_iteration, int wrap)
{
	cv::Mat Result = cv::Mat::zeros(m_Height, m_Width, CV_8UC3);
	//calculate angle and magnitude
		int x, y,index=0;
		// determine motion range:
		float maxx = -999, maxy = -999;
		float minx = 999, miny = 999;
		float maxrad = -1;
		for (y = 0; y < m_Height; y++) {
			for (x = 0; x < m_Width; x++) {
				index = x + y*m_Width;
				float fx = m_gradientFlowUX_Nplus1(index,0);
				float fy = m_gradientFlowUY_Nplus1(index, 0);
				if (unknown_flow(fx, fy))
					continue;
				maxx = __max(maxx, fx);
				maxy = __max(maxy, fy);
				minx = __min(minx, fx);
				miny = __min(miny, fy);
				float rad = sqrt(fx * fx + fy * fy);
				maxrad = __max(maxrad, rad);
			}
		}
#ifdef Debuge
		printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
			maxrad, minx, maxx, miny, maxy);
#endif

		//if (maxmotion > 0) // i.e., specified on commandline
			//maxrad = maxmotion;

		if (maxrad == 0) // if flow == 0 everywhere
			maxrad = 1;

		//if (verbose)
			//fprintf(stderr, "normalizing by %g\n", maxrad);
		uchar *pix = new uchar[3];
		pix[0] = pix[1] = pix[2] = 0;
		for (y = 0; y < m_Height; y++) {
			for (x = 0; x < m_Width; x++) {
				index = x + y*m_Width;
				float fx = m_gradientFlowUX_Nplus1(index, 0);
				float fy = m_gradientFlowUY_Nplus1(index, 0);
				if (unknown_flow(fx, fy)) {
					pix[0] = pix[1] = pix[2] = 0;
				}
				else {
					computeColor(fx / maxrad, fy / maxrad, pix);
				}
				Result.at<cv::Vec3b>(y, x)[0] = pix[0];
				Result.at<cv::Vec3b>(y, x)[1] = pix[1];
				Result.at<cv::Vec3b>(y, x)[2] = pix[2];
			}
		}
		char file_name[50];
		sprintf(file_name, "ConvergePic%d_%d.png",wrap,m_iteration);
		cv::imwrite(file_name, Result);
}
void OpticalFlow::makecolorwheel()
{
	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow 
	//  than between yellow and green)
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	//printf("ncols = %d\n", ncols);
	if (ncols > 60)
		exit(1);
	int i;
	int k = 0;
	for (i = 0; i < RY; i++) setcols(255, 255 * i / RY, 0, k++);
	for (i = 0; i < YG; i++) setcols(255 - 255 * i / YG, 255, 0, k++);
	for (i = 0; i < GC; i++) setcols(0, 255, 255 * i / GC, k++);
	for (i = 0; i < CB; i++) setcols(0, 255 - 255 * i / CB, 255, k++);
	for (i = 0; i < BM; i++) setcols(255 * i / BM, 0, 255, k++);
	for (i = 0; i < MR; i++) setcols(255, 0, 255 - 255 * i / MR, k++);
}
void OpticalFlow::computeColor(float fx, float fy, uchar *pix)
{
	if (ncols == 0)
		makecolorwheel();

	float rad = sqrt(fx * fx + fy * fy);
	float a = atan2(-fy, -fx) / M_PI;
	float fk = (a + 1.0) / 2.0 * (ncols - 1);
	int k0 = (int)fk;
	int k1 = (k0 + 1) % ncols;
	float f = fk - k0;
	//f = 0; // uncomment to see original color wheel
	for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0;
		float col1 = colorwheel[k1][b] / 255.0;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range
		pix[2 - b] = (int)(255.0 * col);
	}
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
void OpticalFlow::Wrapping(Eigen::MatrixXf& m_gradientFlowUX, Eigen::MatrixXf& m_gradientFlowUY, Eigen::MatrixXf& I_x, Eigen::MatrixXf& I_2wrap, Eigen::MatrixXf& I_y, Eigen::MatrixXf& I_Basic)
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
	remap(Illum2, dst, map_x_original, map_y_original, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, I_Basic);
	dst = dst - Illum1;
	Mat2OneRowMatrixf(dst, I_2wrap);

	// By equation 14 in paper<TV-L1 Optical Flow Estimation> to compute the gradient of the Image I1.
	remap(Illum2, dst1, map_x_addone, map_y_original, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst1, WrappedImgFromGradientXADD);
	remap(Illum2, dst, map_x_subone, map_y_original, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, WrappedImgFromGradientXSUB);
	I_x = WrappedImgFromGradientXADD*0.5 - WrappedImgFromGradientXSUB*0.5;

	Eigen::MatrixXf yy = Eigen::Map<Eigen::MatrixXf>(I_x.data(), m_N, 1);




	remap(Illum2, dst1, map_x_original, map_y_addone, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst1, WrappedImgFromGradientYADD);
	remap(Illum2, dst, map_x_original, map_y_subone, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, WrappedImgFromGradientYSUB);
	I_y = WrappedImgFromGradientYADD*0.5 - WrappedImgFromGradientYSUB*0.5;
	cv::Mat mat2 = dst1*0.5 - dst*0.5;



	index = 0;
	float y, q, z = 0.0;
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
