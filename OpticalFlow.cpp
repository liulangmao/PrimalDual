#include "OpticalFlow.h"
#define IterationMax 50 
#define WrapIteration 2
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
	else if (I1.channels() == 1)
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
	m_tau = 1 / std::sqrt(8);
	m_theta = 1 / std::sqrt(8);
	currentLevel = 0;
}

void OpticalFlow::MakeNabla()
{
	/*The order is very important!!*/
	std::vector<Triplet> coefficients;
	m_nabla.resize(m_N * 2, m_N);
	int count = 0;
	for (int r = 0; r <= currentHeight - 1; r++)
	{
		for (int c = 0; c <= currentWidth - 1; c++)
		{
			count = c + r*currentWidth;
			if (c < currentWidth - 1)
			{
				/*if (count >= 2 * m_N - 1 || count + 1 >= m_N)
				{
				printf("boundary error!");
				}*/
				coefficients.push_back(Triplet(count, count, -1));
				coefficients.push_back(Triplet(count, count + 1, 1));
			}
			if (r < currentHeight - 1)
			{
				/*if (count + m_Width*m_Height >= 2 * m_N - 1 || count + m_Width >= m_N)
				{
				printf("boundary error!");
				}*/
				coefficients.push_back(Triplet(count + m_N, count, -1));
				coefficients.push_back(Triplet(count + m_N, count + currentWidth, 1));
			}
		}
	}
	m_nabla.setFromTriplets(coefficients.begin(), coefficients.end());
}
template <typename T>
void OpticalFlow::DebugePrintVector(vector<T> &Myvector)
{
	/*typename std::vector<T>::iterator It1;
	It1 = Myvector.begin();
	for (It1 = Myvector.begin(); It1 != Myvector.end(); ++It1)
	cout << " " << *It1 << endl;*/
	char file_name[50];
	const std::type_info& r = typeid(T);
	const char* name = r.name();
	string s(name);
	if (s == "int")
	{
		printf("vector \n");
		for (int i = 0; i < Myvector.size(); i++)
		{
			printf("vector[%d]:%d \n", i, Myvector[i]);
		}
	}
	else if (s == "float")
	{
		printf("fix code to print vector<float> \n");
	}
	else if (s == "class cv::Mat")
	{
		for (int i = 0; i < Myvector.size(); i++)
		{
			sprintf(file_name, "Pryamid%d.png", i);
			cv::imwrite(file_name, Myvector[i] * 255.0);
		}
	}
}
void OpticalFlow::DebugePrintMatrix(int Width, Eigen::MatrixXf & Matri)
{
	printf("\n ");
	for (int r = 0; r < 5; r++)
	{
		for (int c = 0; c < 5; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf(" ... ");
		for (int c = 15; c < 20; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf("\n");
	}
	for (int r = 5; r < 10; r++)
	{
		for (int c = 0; c < 5; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf(" ... ");
		for (int c = 15; c < 20; c++)
		{
			int x = c + (r)*Width;
			printf(" %.3f ", Matri(x, 0));
		}
		printf("\n");
	}
}
void OpticalFlow::DebugePrintMat(int Width, cv::Mat & Matri)
{
	printf("\n ");
	for (int r = 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
		{
			printf(" %.3f ", Matri.at<float>(r, c));
		}
		printf(" ... ");
		for (int c = 17; c < 20; c++)
		{
			printf(" %.3f ", Matri.at<float>(r, c));
		}
		printf("\n");
	}
	for (int r = 3; r < 10; r++)
	{
		for (int c = 0; c < 3; c++)
		{
			printf(" %.3f ", Matri.at<float>(r, c));
		}
		printf(" ... ");
		for (int c = 17; c < 20; c++)
		{
			printf(" %.3f ", Matri.at<float>(r, c));
		}
		printf("\n");
	}
}
void OpticalFlow::CoarseToFine(int & level, float factor)
{
	vector<int> WidthPryamid, HeightPryamid; float widthtemp = 0.0, heightemp = 0.0;
	WidthPryamid.push_back(Illum1.cols);
	HeightPryamid.push_back(Illum1.rows);
	for (int i = 1; i < level; i++)
	{
		widthtemp = WidthPryamid[i - 1] * factor;
		heightemp = HeightPryamid[i - 1] * factor;
		if ((int)roundf(widthtemp) < 15 || (int)roundf(heightemp) < 15)
		{
			level = i;
			break;
		}
		else{
			if (i == 31)
			{
				WidthPryamid.push_back(22);
				HeightPryamid.push_back(15);
			}
			else
			{
				WidthPryamid.push_back((int)roundf(widthtemp));
				HeightPryamid.push_back((int)roundf(heightemp));
			}
			
			
		}
	}

	for (currentLevel = level - 1; currentLevel >= 0; currentLevel--)
	{
		float scale = factor *(currentLevel - 1);
		int width = WidthPryamid[currentLevel];
		int height = HeightPryamid[currentLevel];

		if (currentLevel == level - 1)
		{
			
			x_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			y_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			px_u_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			px_v_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			py_u_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			py_v_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			Illum1_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			Illum2_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			cv::resize(Illum1, Illum1_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_CUBIC);
			cv::resize(Illum2, Illum2_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_CUBIC);

			for (int r = 0; r < HeightPryamid[currentLevel]; r++)
			{
				for (int c = 0; c < WidthPryamid[currentLevel]; c++)
				{
					x_current.at<float>(r, c) = 0.0;
					y_current.at<float>(r, c) = 0.0;
					px_u_current.at<float>(r, c) = 0.0;
					px_v_current.at<float>(r, c) = 0.0;
					py_u_current.at<float>(r, c) = 0.0;
					py_v_current.at<float>(r, c) = 0.0;
				}
			}
		}
		else
		{
			Illum1_last = Illum1_current;
			Illum2_last = Illum2_current;

			//cv::Mat x_last_temp, y_last_temp;
			x_current.copyTo(x_last);
			y_current.copyTo(y_last);
			px_u_current.copyTo(px_u_last);
			px_v_current.copyTo(px_v_last);
			py_u_current.copyTo(py_u_last);
			py_v_current.copyTo(py_v_last);

			//DebugePrintMat(WidthPryamid[currentLevel + 1], x_last);
			Illum1_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			Illum2_current.create(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			cv::resize(Illum1, Illum1_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_CUBIC);
			cv::resize(Illum2, Illum2_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_CUBIC);
			x_current = cv::Mat::zeros(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			y_current = cv::Mat::zeros(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			px_u_current = cv::Mat::zeros(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			px_v_current = cv::Mat::zeros(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			py_u_current = cv::Mat::zeros(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			py_v_current = cv::Mat::zeros(HeightPryamid[currentLevel], WidthPryamid[currentLevel], CV_32FC1);
			for (int r = 0; r < HeightPryamid[currentLevel]; r++)
			{
				for (int c = 0; c < WidthPryamid[currentLevel]; c++)
				{
					x_current.at<float>(r, c) = 0.0;
					y_current.at<float>(r, c) = 0.0;
					px_u_current.at<float>(r, c) = 0.0;
					px_v_current.at<float>(r, c) = 0.0;
					py_u_current.at<float>(r, c) = 0.0;
					py_v_current.at<float>(r, c) = 0.0;
				}
			}
			//DebugePrintMat(currentWidth, y_current);
			float rescaleWidth = (float)WidthPryamid[currentLevel + 1] / (float)WidthPryamid[currentLevel];
			float rescaleHeigh = (float)HeightPryamid[currentLevel + 1] / (float)HeightPryamid[currentLevel];
			cv::resize(x_last, x_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_NEAREST);

			//DebugePrintMat(currentWidth, x_current);
			cv::resize(y_last, y_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_NEAREST);
			cv::resize(px_u_last, px_u_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_NEAREST);
			cv::resize(px_v_last, px_v_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_NEAREST);
			cv::resize(py_u_last, py_u_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_NEAREST);
			cv::resize(py_v_last, py_v_current, cvSize(WidthPryamid[currentLevel], HeightPryamid[currentLevel]), cv::INTER_NEAREST);

			x_current = x_current / rescaleWidth;
			y_current = y_current / rescaleHeigh;
			
			//DebugePrintMat(currentWidth, y_current);
		}
		m_N = WidthPryamid[currentLevel] * HeightPryamid[currentLevel];
		currentWidth = WidthPryamid[currentLevel];
		currentHeight = HeightPryamid[currentLevel];

		ComputeTv_l1_motion_primal_dual();
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

	Eigen::MatrixXf m_valueX_temp = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_valueY_temp = Eigen::MatrixXf::Zero(m_N, 1);
	//m_valueP = Eigen::MatrixXf::Zero(2 * m_N, m_N);

	Eigen::MatrixXf m_valuePX = Eigen::MatrixXf::Zero(2 * m_N, 1);
	Eigen::MatrixXf m_valuePY = Eigen::MatrixXf::Zero(2 * m_N, 1);

	Eigen::MatrixXf m_WrapX = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_WrapY = Eigen::MatrixXf::Zero(m_N, 1);
	Eigen::MatrixXf m_Wrap_Basic = Eigen::MatrixXf::Zero(m_N, 1);

	Eigen::MatrixXf m_gradientFlowUPX_Nplus1 = Eigen::MatrixXf::Zero(2 * m_N, 1);
	Eigen::MatrixXf m_gradientFlowUPY_Nplus1 = Eigen::MatrixXf::Zero(2 * m_N, 1);

	float L_square = 8;
	m_sigma = 1 / std::sqrt(L_square);

	MakeNabla();//Compute K or K_star here!
	Eigen::MatrixXf WrapNextframe2 = Eigen::MatrixXf::Zero(m_N, 1);

	/*m_valueX_temp is used in maxiteration step as the v vector which is fixed when computing the pho*/

	Eigen::MatrixXf x_Matrix = Eigen::MatrixXf::Zero(currentHeight, currentWidth);
	Eigen::MatrixXf y_Matrix = Eigen::MatrixXf::Zero(currentHeight, currentWidth);
	

	int index = 0;
	for (int r = 0; r < currentHeight; r++)
	{
		for (int c = 0; c < currentWidth; c++)
		{
			index = c + r*currentWidth;
			m_valueX_temp(index, 0) = x_current.at<float>(r, c);
			float qq = m_valueX_temp(0, 0);
			m_valueY_temp(index, 0) = y_current.at<float>(r, c);
			m_valuePX(index, 0) = px_u_current.at<float>(r, c);
			m_valuePY(index, 0) = py_u_current.at<float>(r, c);
			m_valuePX(index + m_N, 0) = px_v_current.at<float>(r, c);
			m_valuePY(index + m_N, 0) = py_v_current.at<float>(r, c);
		}
	}

	/*m_gradientFlowUX is used in maxiteration step as the u vector which is fixed when computing the P*/
	m_gradientFlowUX = m_valueX_temp;
	m_gradientFlowUY = m_valueY_temp;

	for (int i = 0; i < WrapIteration; i++)
	{
		m_gradientFlowUX0 = m_gradientFlowUX;
		m_gradientFlowUY0 = m_gradientFlowUY;
		float yy = m_gradientFlowUX(0,0);
		float qq = m_valueX_temp(0, 0);
		Wrapping(m_gradientFlowUX0, m_gradientFlowUY0, m_WrapX, WrapNextframe2, m_WrapY, m_Wrap_Basic);

		Eigen::MatrixXf pho = Eigen::MatrixXf::Zero(m_N, 1);

		// Steps on page4 in paper<A first-order primal-dual algorithm for convex problems> Algorithm.


		for (int iter = 0; iter < m_iteration; iter++)
		{
			//First to update p
			m_valuePX = m_valuePX + m_sigma*m_nabla*m_valueX_temp;
			//DebugePrintMatrix(currentWidth, m_valuePX);
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
			//DebugePrintMatrix(currentWidth, m_WrapX);
			//DebugePrintMatrix(currentWidth, m_WrapY);
			//DebugePrintMatrix(currentWidth, WrapNextframe2);
			//DebugePrintMatrix(currentWidth, m_gradientFlowUY);
			//DebugePrintMatrix(currentWidth, m_gradientFlowUY0);
			pho = (m_gradientFlowUX - m_gradientFlowUX0).cwiseProduct(m_WrapX) + (m_gradientFlowUY - m_gradientFlowUY0).cwiseProduct(m_WrapY) + WrapNextframe2;
			//DebugePrintMatrix(currentWidth, pho);
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
			//DebugePrintMatrix(currentWidth, m_gradientFlowUX);
			//Steps 4 Update tempU for next iteration
			m_valueX_temp = 2 * m_gradientFlowUX - m_valueX_temp;
			m_valueY_temp = 2 * m_gradientFlowUY - m_valueY_temp;
			//DebugePrintMatrix(currentWidth, m_valueX_temp);
			//
#ifdef Debuge
			if (iter % 10 == 0)
			{
				printf("iteration %d:\n", iter);
				//DebugePrintMatrix(currentWidth,m_gradientFlowUX);
				Eigen::MatrixXf y1 = m_gradientFlowUX.transpose()*m_gradientFlowUX;
				Eigen::MatrixXf y2 = m_gradientFlowUY.transpose()*m_gradientFlowUY;
				//float re = y1(0, 0);
				//float error1 = m_gradientFlowUX(0, 0);
				//float qqerror = m_gradientFlowUX(400, 0);
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
				Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(1, currentWidth);
				Eigen::MatrixXf a = Eigen::MatrixXf::Map(m_WrapX.data(), currentHeight, currentWidth, stride);
				m_WrapX = (m_gradientFlowUY - m_gradientFlowUY0).cwiseProduct(m_WrapY);
				Eigen::MatrixXf b = Eigen::MatrixXf::Map(m_WrapX.data(), currentHeight, currentWidth, stride);
				Eigen::MatrixXf c = Eigen::MatrixXf::Map(m_Wrap_Basic.data(), currentHeight, currentWidth, stride);

				cv::Mat wrapBasic = cv::Mat::zeros(currentHeight, currentWidth, CV_32FC1);
				cv::Mat wrapx = cv::Mat::zeros(currentHeight, currentWidth, CV_32FC1);
				cv::Mat wrapy = cv::Mat::zeros(currentHeight, currentWidth, CV_32FC1);
				cv::eigen2cv(a, wrapx);
				cv::eigen2cv(b, wrapy);
				cv::eigen2cv(c, wrapBasic);
				wrapBasic = wrapBasic + wrapx + wrapy;
				char file_name[50];
				sprintf(file_name, "Wrap%d_%d_%.3f.png", i, iter, Engery(iter, 0));
				cv::imwrite(file_name, wrapBasic*255.0);
			}

#endif
		}//end of iteration
		
		Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(1, currentWidth);
		Eigen::MatrixXf a = Eigen::MatrixXf::Map(m_gradientFlowUX.data(), currentHeight, currentWidth, stride);
		cv::eigen2cv(a, x_current);

		Eigen::MatrixXf b= Eigen::MatrixXf::Map(m_gradientFlowUY.data(), currentHeight, currentWidth, stride);
		cv::eigen2cv(b, y_current);
		index = 0;
		for (int r = 0; r < currentHeight; r++)
		{
			for (int c = 0; c < currentWidth; c++)
			{
				index = c + r*currentWidth;
				px_u_current.at<float>(r, c) = m_valuePX(index, 0);
				py_u_current.at<float>(r, c) = m_valuePY(index, 0);
				px_v_current.at<float>(r, c) = m_valuePX(index + m_N, 0);
				py_v_current.at<float>(r, c) = m_valuePY(index + m_N, 0);
			}
		}
	}


}
void OpticalFlow::ShowFlow(Eigen::MatrixXf& m_gradientFlowUX_Nplus1, Eigen::MatrixXf& m_gradientFlowUY_Nplus1, int m_iteration, int wrap)
{
	cv::Mat Result = cv::Mat::zeros(currentHeight, currentWidth, CV_8UC3);
	//calculate angle and magnitude
	int x, y, index = 0;
	// determine motion range:
	float maxx = -999, maxy = -999;
	float minx = 999, miny = 999;
	float maxrad = -1;
	for (y = 0; y < currentHeight; y++) {
		for (x = 0; x < currentWidth; x++) {
			index = x + y*currentWidth;
			float fx = m_gradientFlowUX_Nplus1(index, 0);
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
	for (y = 0; y < currentHeight; y++) {
		for (x = 0; x < currentWidth; x++) {
			index = x + y*currentWidth;
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
	sprintf(file_name, "ConvergePic%d_%d.png", wrap, m_iteration);
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
	bool is_x_out = false, is_y_out = false;
	if (i >= currentWidth - 1)
	{
		is_x_out = true;
		i = currentWidth - 1;
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


	if (j >= currentHeight - 1)
	{
		is_y_out = true;
		j = currentHeight - 1;
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
	cv::Mat dst, dst1, dst2;
	cv::Mat map_x_original, map_y_original, map_x_addone, map_x_subone, map_y_addone, map_y_subone;
	Eigen::MatrixXf WrappedImgFromGradient, WrappedImgFromGradientXADD, WrappedImgFromGradientXSUB, WrappedImgFromGradientYADD, WrappedImgFromGradientYSUB;


	dst.create(cvSize(currentWidth, currentHeight), Illum2.type());
	map_x_original.create(cvSize(currentWidth, currentHeight), CV_32FC1);
	map_y_original.create(cvSize(currentWidth, currentHeight), CV_32FC1);

	map_x_addone.create(cvSize(currentWidth, currentHeight), CV_32FC1);
	map_y_addone.create(cvSize(currentWidth, currentHeight), CV_32FC1);

	map_x_subone.create(cvSize(currentWidth, currentHeight), CV_32FC1);
	map_y_subone.create(cvSize(currentWidth, currentHeight), CV_32FC1);

	WrappedImgFromGradient = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientXADD = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientXSUB = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientYADD = Eigen::MatrixXf::Zero(m_N, 1);
	WrappedImgFromGradientYSUB = Eigen::MatrixXf::Zero(m_N, 1);
	/// Update m_gradientFlowUX & m_gradientFlowUY. Then apply remap
	int index = 0; int row = 0, col = 0;
	for (int rr = 0; rr < currentHeight; rr++)
	{
		for (int cc = 0; cc < currentWidth; cc++)
		{
			index = cc + rr*currentWidth;
			row= m_gradientFlowUY(index, 0) + rr;
			col = m_gradientFlowUY(index, 0) + cc;
			bool is_boundary = Isboundary(col, row);
			map_x_original.at<float>(rr, cc) = col;
			map_y_original.at<float>(rr, cc) = row;
			row = row + 1;
			col = col + 1;
			is_boundary = Isboundary(col, row);
			map_x_addone.at<float>(rr, cc) = col;
			map_y_addone.at<float>(rr, cc) = row;
			row = row - 2;
			col = col - 2;
			is_boundary = Isboundary(col, row);
			map_x_subone.at<float>(rr, cc) = col;
			map_y_subone.at<float>(rr, cc) = row;
		}
	}

	/*Debug: */
	/*ifstream in_stream;
	in_stream.open("I1.txt");
	string line;
	int  ccol = 0;
	row = 0;
	if (in_stream.is_open())
	{
		while (getline(in_stream, line))
		{
			(Illum1_current).at<float>(row, ccol) = std::stof(line);
			cout << (Illum1_current).at<float>(row, ccol) << endl;
			ccol++;
			if (ccol == currentWidth)
			{
				ccol = 0; row++;
			}
			line.clear();
		}
		in_stream.close();
	}
	
	ifstream in_stream2;
	in_stream2.open("I2.txt");
	row = 0, ccol = 0;
	if (in_stream2.is_open())
	{
		while (getline(in_stream2, line))
		{
			(Illum2_current).at<float>(row, ccol) = std::stof(line);
			cout << (Illum2_current).at<float>(row, ccol) << endl;
			ccol++;
			if (ccol == currentWidth)
			{
				ccol = 0; row++;
			}
			line.clear();
		}
		in_stream2.close();
	}*/
	
	//DebugePrintMat(currentWidth, Illum1_current);
	
	//DebugePrintMat(currentWidth, Illum2_current);
	//DebugePrintMat(currentWidth, map_x_original);
	//DebugePrintMat(currentWidth, map_y_original);
	remap((Illum2_current), dst, map_x_original, map_y_original, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, I_Basic);
	//DebugePrintMat(currentWidth, dst);
	dst = dst - (Illum1_current);
	Mat2OneRowMatrixf(dst, I_2wrap);
	//DebugePrintMatrix(currentWidth, I_2wrap);


	// By equation 14 in paper<TV-L1 Optical Flow Estimation> to compute the gradient of the Image I1.
	remap((Illum2_current), dst1, map_x_addone, map_y_original, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst1, WrappedImgFromGradientXADD);
	remap((Illum2_current), dst, map_x_subone, map_y_original, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, WrappedImgFromGradientXSUB);
	I_x = WrappedImgFromGradientXADD*0.5 - WrappedImgFromGradientXSUB*0.5;


	//DebugePrintMatrix(currentWidth, I_x);
	Eigen::MatrixXf yy = Eigen::Map<Eigen::MatrixXf>(I_x.data(), m_N, 1);




	remap((Illum2_current), dst1, map_x_original, map_y_addone, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst1, WrappedImgFromGradientYADD);
	remap((Illum2_current), dst, map_x_original, map_y_subone, CV_INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	Mat2OneRowMatrixf(dst, WrappedImgFromGradientYSUB);
	I_y = WrappedImgFromGradientYADD*0.5 - WrappedImgFromGradientYSUB*0.5;
	cv::Mat mat2 = dst1*0.5 - dst*0.5;
	//DebugePrintMatrix(currentWidth, I_y);


	index = 0;
	float y, q, z = 0.0;
	for (int r = 0; r < dst.rows; r++)
	{
		for (int c = 0; c < dst.cols; c++)
		{
			index = c + r*currentWidth;
			if (Isboundary(c, r))
			{
				I_2wrap(index, 0) = 0;
				I_x(index, 0) = 0;
				I_y(index, 0) = 0;
			}
		}
	}
	//DebugePrintMatrix(currentWidth, I_x);

}
OpticalFlow::~OpticalFlow()
{
}
