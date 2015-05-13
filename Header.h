#pragma once
#define Debuge 0
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
typedef Eigen::SparseMatrix<float> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<float> T;


enum class Application { PrimalDual_ROF, OpticalFlow_ };
enum class Method { HuberRof, WeightedHuberRof, TV_L1 };