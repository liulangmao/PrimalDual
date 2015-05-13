#include "PrimalDualAl.h"


PrimalDualAl::PrimalDualAl(cv::Mat & inputImg, int steps, float afa, float lamda, Method method)
{
	m_InputImg = inputImg;
	m_NumStep = steps;
	m_delta = afa;
	m_gamma = lamda;
	m_method = method;
	m_width = m_InputImg.cols;
	m_height = m_InputImg.rows;
}


PrimalDualAl::~PrimalDualAl()
{
}

void PrimalDualAl::Denoise()
{
	SetParas();
	PreCompute();
	float denominator_result = 1 + m_tau*m_gamma;
	Eigen::MatrixXf convergeValue = Eigen::MatrixXf::Zero(m_NumStep+1,1);
    //Initalization
	convergeValue(0, 0) = FirstValue() + (m_gamma/2)*DualValue();
	printf("\n[Initalization has done! the distance is %f]\n", convergeValue(0, 0));
	IterConvergence(convergeValue);
}
void PrimalDualAl::SetParas()
{
	if (m_method == Method::HuberRof)
	{
		m_L = sqrt(8);
		m_mu = 2 * sqrt(m_gamma * m_delta) / m_L;
		m_tau = m_mu / (2 * m_gamma);
		m_sigma = m_mu / (2 * m_delta);
		m_theta = 1 / (1 + m_mu);
	}
	else{
		printf("Not HuberRof!\n");
	}

}
void PrimalDualAl::testX_Y(int rtest, int ctest)
{
	Eigen::MatrixXf test= m_YFirstAndSeconderivateData*m_XdimenData;
	printf("\nTest: Sparse Matrix value\n");
	for (int r = -1; r < 2; r++){
		for (int c = -1; c < 2; c++){
			printf("%d ", (int)m_InputImg.at<float>(rtest + r, ctest + c));
		}
		printf("\n");
	}
    printf("firstd[(uj+1,i)-(uj,i)]=:%d,Secondd[(uj,i+1)-(uj,i)]=:%d\n", (int)test(ctest + rtest*m_width, 0), (int)test(ctest + rtest*m_width+m_width*m_height, 0));
	printf("Sucessfully!\n");
}
void PrimalDualAl::PreCompute()
{
	m_XdimenData.resize(m_width*m_height,1);
	int count = 0;
	for (int r = 0; r < m_height; r++)
	{
		for (int c = 0; c < m_width; c++)
		{
			m_XdimenData(count,0) = m_InputImg.at<float>(r, c);
			count++;
		}
	}
	m_result = m_XdimenData;
	m_result_line = m_result;
	std::vector<T> coefficients;
	m_YFirstAndSeconderivateData.resize(m_width*m_height * 2, m_width*m_height);
	count = 0;
	for (int r = 0; r < m_height-1; r++)
	{
		for (int c = 0; c < m_width-1; c++)
		{
			count = c + r*m_width;
			coefficients.push_back(T(count, count, -1));
			coefficients.push_back(T(count, count + m_width, 1));
			coefficients.push_back(T(count + m_width*m_height, count, -1));
			coefficients.push_back(T(count + m_width*m_height, count + 1, 1));
			/*m_YFirstAndSeconderivateData(count, count) = -1;
			m_YFirstAndSeconderivateData(count, count + m_width) = 1;
			m_YFirstAndSeconderivateData(count + m_width*m_height, count) = -1;
			m_YFirstAndSeconderivateData(count + m_width*m_height, count + 1) = 1;*/
		}
	}
	m_YFirstAndSeconderivateData.setFromTriplets(coefficients.begin(), coefficients.end());
	m_YIterCoef_line.resize(m_width*m_height * 2, 1);
	m_OutputImg.create(m_height,m_width,CV_32FC1);
#if Debuge==1
	testX_Y(1, 1);
	testX_Y(m_width - 2, m_width - 2);
	//testX_Y(0, 0);//sucessfully
	//testX_Y(m_width - 1, m_width - 1);//Novalue
#endif
}

float PrimalDualAl::FirstValue()
{
	Eigen::MatrixXf Firstdata = m_YFirstAndSeconderivateData*m_result;
	int sum = m_width*m_height;
	Eigen::MatrixXf Firstemp = Eigen::MatrixXf::Zero(sum, 1);
	float partOne = 0, partTwo = 0;
	for (int n = 0; n < sum;n++)
	{
		Firstemp(n, 0) = sqrt(Firstdata(n, 0)*Firstdata(n, 0) + Firstdata(n + sum, 0)*Firstdata(n + sum, 0));
		if (Firstemp(n, 0) <= m_delta)
		{
			partOne += (Firstemp(n, 0)*Firstemp(n, 0)) / (2 * m_delta);
		}
		else
		{
			partTwo += Firstemp(n, 0)-(m_delta/2.0);
		}
	}
	return partOne + partTwo;
}
float PrimalDualAl::DualValue()
{
	Eigen::MatrixXf Secondata = m_result - m_XdimenData;
	Eigen::MatrixXf tempTran = Secondata.transpose();
	Eigen::MatrixXf y = tempTran*Secondata;
	//float y = (Secondata.transpose)*Secondata;
	return y(0,0);
}
void PrimalDualAl::IterConvergence(Eigen::MatrixXf & convergeValue)
{
	/********************************************
	p or yN or yN+1 in paper equals m_YIterCoef_line 
	u or xN or xN+1 in paper equals m_result which contains our final result means uN or uN+1
	xN+1^ or xN^ in paper equals m_result_line means temporary update valuable.
	g(i,j) in paper equals m_Xdata_line 
	K in paper eqauls m_YFirstAndSeconderivateData
	Kstar in paper eqauls m_divop
	*******************************************/
	//m_divop = m_YFirstAndSeconderivateData.conjugate();
	m_divop = m_YFirstAndSeconderivateData.transpose();
	for (int stepN = 1; stepN < m_NumStep + 1; stepN++)
	{
		///update the p^
		m_YIterCoef_line = m_sigma*m_YFirstAndSeconderivateData*m_result_line + m_YIterCoef_line;
		/**m_YIterCoef_line_temp.addTo(m_YIterCoef_line); //take attention! one is sparseMatrix and other is DenseMatrix. They cannot add together.**/
		m_YIterCoef_line = m_YIterCoef_line / (1 + m_sigma * m_delta);
		//compute pN+1 with P^
		int sum = m_width*m_height;
		Eigen::MatrixXf Ytemp = Eigen::MatrixXf::Zero(2*sum, 1);
		float partOne = 0, partTwo = 0;
		for (int n = 0; n < sum; n++)
		{
			Ytemp(n, 0) = sqrt(m_YIterCoef_line(n, 0)*m_YIterCoef_line(n, 0) + m_YIterCoef_line(n + sum, 0)*m_YIterCoef_line(n + sum, 0));
			if (Ytemp(n, 0) <= 1)
			{
				Ytemp(n, 0) = 1;
			}
			Ytemp(n + sum, 0) = Ytemp(n, 0);
			m_YIterCoef_line(n, 0) = m_YIterCoef_line(n, 0) / Ytemp(n, 0);
			m_YIterCoef_line(n+sum, 0) = m_YIterCoef_line(n+sum, 0) / Ytemp(n+sum, 0);
		}
        //update the u^
		Eigen::MatrixXf XNcopy = m_result;
		//Eigen::MatrixXf Xtemp = m_divop*m_YIterCoef_line;
		Eigen::MatrixXf Xtemp = m_result - m_tau*m_divop*m_YIterCoef_line;
		m_result = (Xtemp + m_tau*m_gamma*m_XdimenData) / (1 + m_tau*m_gamma);
		//update the x^
		m_result_line = m_result + m_theta*(m_result - XNcopy);
#if PrintConValue==1
		float testFirstValue = FirstValue();
		float testSecondValue = DualValue();
		//printf("[%d loop: testFirstValue is %f;testSecondValue is %f\n", stepN,testFirstValue, testSecondValue);
#endif 
		convergeValue(stepN, 0) = FirstValue() + (m_gamma / 2)*DualValue();
		printf("[%d loop has done! the distance is %f]\n", stepN, convergeValue(stepN, 0));
		int countImg = 0;
		for (int i = 0; i < m_height; i++)
		{
			for (int j = 0; j < m_width; j++)
			{
				countImg = j + i*m_width;
				m_OutputImg.at<float>(i, j) = m_result(countImg, 0);
			}
		}
		char file_name[50];
		sprintf(file_name, "ConvergePic%d.png", stepN);
		//m_OutputImg.convertTo(m_OutputImg, CV_32F, 255, 0);
		cv::imwrite(file_name, m_OutputImg*255.0);
	}
}
