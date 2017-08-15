#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
	// Setup and initialize an RMSE to be returned in case of errors
	VectorXd err_rmse(4);
	err_rmse << -1, -1, -1, -1;


	// Check if the estimation and ground truth are of same size
	if ( estimations.size() != ground_truth.size() )
	{
		cout << "[Error] Estimations and Ground truth measurements are of different size!" << endl;
		cout << "Cannot calculate RMSE. RMSE set to -1. " << endl;

		return err_rmse;
	}
	else if ( estimations.size() == 0 )
	{
		cout << "[Error] Estimations are of size 0!" << endl;
		cout << "Cannot calculate RMSE. RMSE set to -1." << endl;

		return err_rmse;
	}

	// Setup the RMSE vector and initialize to 0
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// Cumulative squared residuals
	for (int i=0;i<estimations.size();i++)
	{
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	// Calculate the mean
	rmse /= estimations.size();

	// Calculate the square root
	rmse = rmse.array().sqrt();

	//return
	return rmse;
  
}