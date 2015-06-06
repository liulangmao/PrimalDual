#pragma once
#define Debuge 0
#undef DEBUG_VERBOSE 
#define DEBUG_VERBOSE 1		 

#if DEBUG_VERBOSE		
#define debug_printf printf		
#else		
#define debug_printf (void)		 
#endif		 

using namespace std;
#define PrintConValue 1
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <typeinfo>
typedef Eigen::SparseMatrix<float> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<float> Triplet;
#include <iostream>
#include <fstream>

enum class Application { PrimalDual_ROF, OpticalFlow_ };
enum class Method { HuberRof, WeightedHuberRof, TV_L1 };