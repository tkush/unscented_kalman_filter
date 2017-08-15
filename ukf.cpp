#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

#include <fstream>

#define TARGET_RMSE_P (0.11)

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void NormalizeToPi(double &val)
{
	while (val > M_PI) 
	{
		//cout << "Original: "<< val;
		val -= 2.*M_PI;
		//cout << " New: " << val << endl;
	}
	while (val < -M_PI) 
	{
		//cout << "Original: "<< val;
		val += 2.*M_PI;
		//cout << " New: " << val << endl;
	}
}
/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {

	// measurement index
	idx_ = 0;

	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// file handler for NIS
	if (use_radar_)
		nis_radar_ = fopen("radar_NIS.txt","w");
	else nis_radar_ = NULL;

	if (use_laser_)
		nis_lidar_ = fopen("lidar_NIS.txt","w");
	else nis_lidar_ = NULL;
	
	// initial state vector
	x_ = VectorXd(5);

	// initial covariance matrix
	P_ = MatrixXd(5, 5);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 4;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 1;

	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;

	// Is initialized?
	is_initialized_ = false;

	// States dimension
	n_x_ = 5;

	// Augmented states dimension
	n_aug_ = 7;

	// Weights for sigma points
	weights_ = VectorXd(2*n_aug_+1);

	// Sigma prediction matrix
	Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

	// Spreading function
	lambda_ = 3 - n_aug_;

}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
	if (!is_initialized_) 
	{
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) 
		{
			double rho = meas_package.raw_measurements_(0);
			double phi = meas_package.raw_measurements_(1);
			double px, py;

			if ( rho < 1e-5 )
			{
				// target is very close to radar
				// initialize px, py to be target rmse
				px = TARGET_RMSE_P;
				py = TARGET_RMSE_P;
			}		
			else
			{
				px = rho * cos (phi);
				py = rho * sin (phi);
			}
			// States: px, py, v (mag), yaw, yaw_rate
			x_ << px, py, 0, 0, 0;
			// Proess noise covariance
			P_ << std_radr_*std_radr_, 0, 0, 0, 0,
				  0, std_radr_*std_radr_, 0, 0, 0,
				  0, 0, std_radr_*std_radr_, 0, 0,
				  0, 0, 0, 1, 0,
				  0, 0, 0, 0, 1;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) 
		{
			// raw measurements contains px, py. Initialize v, yaw, yaw_rate to 0?
			x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
			// Proess noise covariance
			P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
				  0, std_laspy_*std_laspy_, 0, 0, 0,
				  0, 0, 1, 0, 0,
				  0, 0, 0, 1, 0,
				  0, 0, 0, 0, 1;
		}

		prev_timestamp_ = meas_package.timestamp_;
		
		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}
	else
	{
		long delta_t = meas_package.timestamp_ - prev_timestamp_;
		double dt = delta_t / 1.e6;
		prev_timestamp_ = meas_package.timestamp_;

		// do not make a prediction if dt is small
		if ( dt > 1e-5 )
		{
			Prediction(dt);
			if ( meas_package.sensor_type_ == MeasurementPackage::LASER )
				UpdateLidar(meas_package);
			else
				UpdateRadar(meas_package);
			idx_++;
		}
	}
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) 
{
	double spread = sqrt( lambda_ + n_aug_ );

	// define the augmented state vector
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(n_x_) << x_;
	x_aug.tail(n_aug_- n_x_) << 0,0;

	// Define the augmented sigma matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
	
	// Define the augmented covariance matrix
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_,n_x_) = std_a_*std_a_;
	P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

	// square root of P_aug
	MatrixXd A = P_aug.llt().matrixL();

	// First column is just the state vector
	Xsig_aug.col(0) << x_aug;
	// Fill in the rest
	for (int i=0;i<n_aug_;i++)
	{
		Xsig_aug.col(i+1) = x_aug + spread * A.col(i);
		Xsig_aug.col(i+n_aug_+1) = x_aug - spread * A.col(i);
	}

	// Predict the sigma points
	VectorXd temp = VectorXd(n_x_);
	for (int i=0; i< Xsig_aug.cols(); i++)
	{
		double vk = Xsig_aug.col(i)(2);
		double psik = Xsig_aug.col(i)(3);
		double psi_d_k = Xsig_aug.col(i)(4);
		double nu_ak = Xsig_aug.col(i)(5);
		double nu_psik = Xsig_aug.col(i)(6);

		if ( fabs(psi_d_k) < 1e-5)
		{
			temp(0) = vk * cos( psik ) * delta_t + 0.5*delta_t*delta_t*cos( psik ) * nu_ak;
			temp(1) = vk * sin( psik ) * delta_t + 0.5*delta_t*delta_t*sin( psik ) * nu_ak;
			temp(2) = delta_t * nu_ak;
			temp(3) = 0.5*delta_t*delta_t*nu_psik;
			temp(4) = delta_t * nu_psik;
		}
		else
		{
			double theta = psik + psi_d_k* delta_t;
			temp(0) = (vk/psi_d_k) * ( sin( theta ) - sin( psik ) ) + 0.5*delta_t*delta_t*cos( psik ) * nu_ak;
			temp(1) = (vk/psi_d_k) * (-cos( theta ) + cos( psik ) ) + 0.5*delta_t*delta_t*sin( psik ) * nu_ak;
			temp(2) = delta_t * nu_ak;
			temp(3) = psi_d_k*delta_t + 0.5*delta_t*delta_t*nu_psik;
			temp(4) = delta_t * nu_psik;
		}

		Xsig_pred_.col(i) = Xsig_aug.col(i).head(5) + temp;
	}

	// Set the state vector to 0
	x_.fill(0.0);

	//set weights
	int n_sig = 2*n_aug_ + 1;
	for (int i=0;i<n_sig;i++)
	{
		if (i==0)
			weights_(i) = lambda_/(lambda_ + n_aug_);
		else
			weights_(i) = 1./(2*(lambda_ + n_aug_));
	}
    
	// predict state mean
	for (int i=0;i<n_sig;i++)
	{
		x_ = x_ + weights_(i) * Xsig_pred_.col(i);
	}

	//predict state covariance matrix
	P_.fill(0.0);
	// VectorXd temp = VectorXd(n_x_);
	for (int i=0;i<n_sig;i++)
	{
		temp = Xsig_pred_.col(i) - x_;
		P_ +=  weights_(i)* temp * temp.transpose();
	}
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
	// number of states for lidar
	int n_z = 2;

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	// Measured state
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

	// Matrix for holding sigma points transformed to measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z,n_z);

	// transform sigma points into measurement space
	for (int i=0;i<2*n_aug_+1;i++)
	{
		double px = Xsig_pred_.col(i)(0);
		double py = Xsig_pred_.col(i)(1);

		Zsig.col(i)(0) = px;
		Zsig.col(i)(1) = py;
	}

	//calculate mean predicted measurement
	z_pred.fill(0.);
	for (int i=0;i<2*n_aug_+1;i++)
	{
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//calculate measurement covariance matrix S
	S.fill(0.);
	VectorXd temp = VectorXd(n_z);
	MatrixXd R = MatrixXd(n_z, n_z);
	R.fill(0.);
	R(0,0) = std_laspx_*std_laspx_;
	R(1,1) = std_laspy_*std_laspy_;
	for (int i=0;i<2*n_aug_+1;i++)
	{
		temp = Zsig.col(i) - z_pred;
		NormalizeToPi(temp(1));
		S = S + weights_(i) * temp * temp.transpose();
	}
	S = S + R;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.);
	VectorXd diff_x = VectorXd(n_x_);
	VectorXd diff_z = VectorXd(n_z);
	for (int i=0;i<2*n_aug_ + 1;i++)
	{
		diff_x = Xsig_pred_.col(i) - x_;
		diff_z = Zsig.col(i) - z_pred;
		
		Tc = Tc + weights_(i)*diff_x*diff_z.transpose();
	}

	//calculate Kalman gain K;
	MatrixXd S_inv = S.inverse();
	MatrixXd K = MatrixXd(n_x_, n_z);
	K = Tc * S_inv;

	//update state mean and covariance matrix
	x_ = x_ + K * (z - z_pred);
	P_ = P_ - K*S*K.transpose();

	// calculate NIS
	double epsilon;
	VectorXd diff_meas = VectorXd(n_z);
	diff_meas = z - z_pred;
	epsilon = diff_meas.transpose() * S_inv * diff_meas;
	fprintf(nis_lidar_,"%d,%lf\n",idx_,epsilon);
}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {

	// number of states for radar
	int n_z = 3;

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	// Measured state
	VectorXd z = VectorXd(n_z);
	z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

	// Matrix for holding sigma points transformed to measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z,n_z);

	// transform sigma points into measurement space
	for (int i=0;i<2*n_aug_+1;i++)
	{
		double px = Xsig_pred_.col(i)(0);
		double py = Xsig_pred_.col(i)(1);
		double v = Xsig_pred_.col(i)(2);
		double phi = Xsig_pred_.col(i)(3);
		double sq_sum = px*px + py*py;

		Zsig.col(i)(0) = sqrt(sq_sum);

		if ( fabs(px) < 1e-5 && fabs(py) < 1e-5)
			Zsig.col(i)(1) = 0;
		else
			Zsig.col(i)(1) = atan2(py, px);

		if ( sq_sum < 1e-5 )
			Zsig.col(i)(2) = 0;
		else
			Zsig.col(i)(2) = (px*cos(phi)*v+py*sin(phi)*v)/Zsig.col(i)(0);
	}

	//calculate mean predicted measurement
	z_pred.fill(0.);
	for (int i=0;i<2*n_aug_+1;i++)
	{
		z_pred = z_pred + weights_(i) * Zsig.col(i);
	}

	//calculate measurement covariance matrix S
	S.fill(0.);
	VectorXd temp = VectorXd(n_z);
	MatrixXd R = MatrixXd(n_z, n_z);
	R.fill(0.);
	R(0,0) = std_radr_*std_radr_;
	R(1,1) = std_radphi_*std_radphi_;
	R(2,2) = std_radrd_*std_radrd_;
	for (int i=0;i<2*n_aug_+1;i++)
	{
		temp = Zsig.col(i) - z_pred;
		NormalizeToPi(temp(1));
		S = S + weights_(i) * temp * temp.transpose();
	}
	S = S + R;

	//create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);

	//calculate cross correlation matrix
	Tc.fill(0.);
	VectorXd diff_x = VectorXd(n_x_);
	VectorXd diff_z = VectorXd(n_z);
	for (int i=0;i<2*n_aug_ + 1;i++)
	{
		diff_x = Xsig_pred_.col(i) - x_;
		diff_z = Zsig.col(i) - z_pred;
		NormalizeToPi(diff_x(3));
		NormalizeToPi(diff_z(1));

		Tc = Tc + weights_(i)*diff_x*diff_z.transpose();
	}

	//calculate Kalman gain K;
	MatrixXd S_inv = S.inverse();
	MatrixXd K = MatrixXd(n_x_, n_z);
	K = Tc * S_inv;

	//update state mean and covariance matrix
	x_ = x_ + K * (z - z_pred);
	P_ = P_ - K*S*K.transpose();

	// calculate NIS
	double epsilon;
	VectorXd diff_meas = VectorXd(n_z);
	diff_meas = z - z_pred;
	epsilon = diff_meas.transpose() * S_inv * diff_meas;	
	fprintf(nis_radar_,"%d,%lf\n",idx_,epsilon);
}
