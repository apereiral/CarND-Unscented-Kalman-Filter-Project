#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initialize state vector dimension
  n_x_ = 5;

  // initialize augmented state dimension
  n_aug_ = 7;

  // initialize sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial augmented state vector
  x_aug_ = VectorXd(n_aug_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // initial augmented covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  // initial radar measurement noise covariance matrix
  R_radar_ = MatrixXd(3, 3);

  // initial lidar measurement noise covariance matrix
  R_lidar_ = MatrixXd(2, 2);

  // initial sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);
  
  // initial lidar measurement function matrix
  H_lidar_ = MatrixXd(2, n_x_);

  // initial weights vector
  weights_ = VectorXd(2*n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

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

  // if this is false, state vector hasn't been initialized yet
  is_initialized_ = false;

  // covariance matrix initialization
  P_ << 1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1;

  // radar measurement noise covariance matrix initialization
  R_radar_ << std_radr_*std_radr_, 0, 0,
			  0, std_radphi_*std_radphi_, 0,
			  0, 0, std_radrd_*std_radr_;

  // lidar measurement noise covariance matrix initialization
  R_lidar_ << std_laspx_*std_laspx_, 0,
			  0, std_laspy_*std_laspy_;

  // lidar measurement function matrix
  H_lidar_.fill(0.0);
  H_lidar_(0, 0) = 1;
  H_lidar_(1, 1) = 1;

  // augmented state and covariance matrix initialization
  x_aug_.fill(0.0);
  P_aug_.fill(0.0);

  // weights vector
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for(int i = 1; i < 2*n_aug_ + 1; i++){
	  weights_(i) = 0.5/(lambda_ + n_aug_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  Process and fuse lidar and radar measurements.
  */
  if(!is_initialized_){
	// initialize state vector with first measurement
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		// convert radar from polar to cartesian coordinates and initialize state
		double rho = meas_package.raw_measurements_[0];
		double phi = meas_package.raw_measurements_[1];
		double rho_dot = meas_package.raw_measurements_[2];
		x_ << rho*cos(phi), rho*sin(phi), 0, 0, 0;
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		// initialize state
		x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
	}

    // record timestamp and mark UKF as initialized
	time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // calculate time since last measurement
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;

  //cout << x_ << "\n" << endl;
  /// Prediction step
  Prediction(dt);
  
  /// Update step
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
	// lidar measurement
	UpdateLidar(meas_package);
  }
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
	// radar measurement
	UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. 
  Predict sigma points, the state vector, and the state covariance matrix.
  */
  // augmented covariance matrix and state vector initialization
  P_aug_.topLeftCorner(n_x_, n_x_) = P_;
  P_aug_(n_x_,n_x_) = std_a_*std_a_;
  P_aug_(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;
  x_aug_.head(n_x_) = x_;

  // calculate A that solves A*A.transpose() = P_aug_
  MatrixXd A = P_aug_.llt().matrixL();

  /// Calculate and predict sigma point matrix
  // generate initial sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ + 1);
  Xsig_aug.col(0) = x_aug_;
  for(int i = 1; i <= n_aug_; i++){
	Xsig_aug.col(i) = x_aug_ + sqrt(lambda_ + n_aug_)*A.col(i - 1);
	Xsig_aug.col(n_aug_ + i) = x_aug_ - sqrt(lambda_ + n_aug_)*A.col(i - 1);
  }
  // predict sigma points from process function
  for(int j = 0; j < 2*n_aug_ + 1; j++){
	float p_x = Xsig_aug(0, j);
	float p_y = Xsig_aug(1, j);
	float v = Xsig_aug(2, j);
	float yaw = Xsig_aug(3, j);
	float yawd = Xsig_aug(4, j);
	float nu_a = Xsig_aug(5, j);
	float nu_ydd = Xsig_aug(6, j);
    Xsig_pred_(2, j) = v + delta_t*nu_a;
    Xsig_pred_(3, j) = yaw + delta_t*yawd + 0.5*delta_t*delta_t*nu_ydd;
	Xsig_pred_(4, j) = yawd + delta_t*nu_ydd;
    // avoid division by 0
	if(fabs(yawd) > 0.001){
        Xsig_pred_(0, j) = p_x + v*(sin(yaw + yawd*delta_t) - sin(yaw))/yawd + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
        Xsig_pred_(1, j) = p_y + v*(-cos(yaw + yawd*delta_t) + cos(yaw))/yawd + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
    } else{
        Xsig_pred_(0, j) = p_x + v*cos(yaw)*delta_t + 0.5*delta_t*delta_t*cos(yaw)*nu_a;
        Xsig_pred_(1, j) = p_y + v*sin(yaw)*delta_t + 0.5*delta_t*delta_t*sin(yaw)*nu_a;
    }
  }

  /// Calculate predicted state and covariance matrix
  // predicted mean state
  x_ = weights_(0)*Xsig_pred_.col(0);
  for(int j = 1; j < 2*n_aug_ + 1; j++){
	x_ += weights_(j)*Xsig_pred_.col(j);
  }
  //normalize yaw angle
  x_(3) = NormAngle(x_(3));
  // predicted covariance matrix
  VectorXd x_diff = Xsig_pred_.col(0) - x_;
  //normalize yaw angle
  x_diff(3) = NormAngle(x_diff(3));
  P_ = weights_(0)*(x_diff)*(x_diff.transpose());
  for(int j = 1; j < 2*n_aug_ + 1; j++){
	x_diff = Xsig_pred_.col(j) - x_;
	//normalize yaw angle
	x_diff(3) = NormAngle(x_diff(3));
	P_ += weights_(j)*(x_diff)*(x_diff.transpose());
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  It also calculates the lidar NIS.
  */
  /// Calculate predicted measured state and covariance matrix
  // read new lidar measurements
  VectorXd z = meas_package.raw_measurements_;
  // apply linear KF equations
  VectorXd z_pred = H_lidar_*x_;
  VectorXd y = z - z_pred;
  MatrixXd PHt = P_*H_lidar_.transpose();
  MatrixXd S = H_lidar_*PHt + R_lidar_;
  MatrixXd S_inv = S.inverse();
  MatrixXd K = PHt*S_inv;

  x_ += K*y;
  //normalize yaw angle
  x_(3) = NormAngle(x_(3));
  P_ -= K*H_lidar_*P_;

  NIS_laser_ = ((z - z_pred).transpose())*(S_inv)*(z - z_pred);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  It also calculates the radar NIS.
  */
  /// Calculate predicted measured state and covariance matrix
  // predicted measured sigma points matrix
  MatrixXd Zsig_pred = MatrixXd(3, 2*n_aug_ + 1);
  for(int j = 0; j < 2*n_aug_ + 1; j++){
	  Zsig_pred(0, j) = sqrt(Xsig_pred_(0, j)*Xsig_pred_(0, j) + Xsig_pred_(1, j)*Xsig_pred_(1, j));
	  Zsig_pred(1, j) = atan2(Xsig_pred_(1, j), Xsig_pred_(0, j));
	  Zsig_pred(2, j) = (Xsig_pred_(0, j)*cos(Xsig_pred_(3, j)) + 
						Xsig_pred_(1, j)*sin(Xsig_pred_(3, j)))*Xsig_pred_(2, j)/Zsig_pred(0, j);
  }
  // predicted mean state
  VectorXd z_pred = weights_(0)*Zsig_pred.col(0);
  for(int j = 1; j < 2*n_aug_ + 1; j++){
	z_pred += weights_(j)*Zsig_pred.col(j);
  }
  // predicted measurement covariance matrix
  VectorXd z_diff = Zsig_pred.col(0) - z_pred;
  //normalize phi angle
  z_diff(1) = NormAngle(z_diff(1));
  MatrixXd S = weights_(0)*(z_diff)*(z_diff.transpose());
  for(int j = 1; j < 2*n_aug_ + 1; j++){
	z_diff = Zsig_pred.col(j) - z_pred;
	//normalize phi angle
	z_diff(1) = NormAngle(z_diff(1));
	S += weights_(j)*(z_diff)*(z_diff.transpose());
  }
  S += R_radar_;

  // read new lidar measurements
  VectorXd z = meas_package.raw_measurements_;

  /// Calculate cross correlation matrix and Kalman gain matrix
  VectorXd x_diff = Xsig_pred_.col(0) - x_;
  //normalize yaw angle
  x_diff(3) = NormAngle(x_diff(3));
  z_diff = Zsig_pred.col(0) - z_pred;
  //normalize phi angle
  z_diff(1) = NormAngle(z_diff(1));
  // calculate cross correlation matrix
  MatrixXd Tc = weights_(0)*(x_diff)*(z_diff).transpose();
  for(int j = 1; j < 2*n_aug_ + 1; j++){
	x_diff = Xsig_pred_.col(j) - x_;
	//normalize yaw angle
	x_diff(3) = NormAngle(x_diff(3));
	z_diff = Zsig_pred.col(j) - z_pred;
	//normalize phi angle
	z_diff(1) = NormAngle(z_diff(1));
	Tc += weights_(j)*(x_diff)*(z_diff.transpose());
  }
  // calculate Kalman gain matrix
  MatrixXd S_inv = S.inverse();
  MatrixXd K = Tc*S_inv;

  /// Update state vector and covariance matrix
  // update mean state vector
  z_diff = z - z_pred;
  //normalize phi angle
  z_diff(1) = NormAngle(z_diff(1));
  x_ = x_ + K*(z_diff);
  //normalize yaw angle
  x_(3) = NormAngle(x_(3));
  // update state covariance matrix
  P_ = P_ - K*S*K.transpose();

  /// Calculate NIS for consistency check
  NIS_radar_ = (z_diff.transpose())*S_inv*z_diff;
}

double UKF::NormAngle(double angle){
  /**
  Normalize given angle between -pi and pi.
  */
  if(angle > M_PI || angle < -M_PI){
	int sign = (angle > 0) - (angle < 0);
	angle += sign*M_PI;
	angle = fmod(angle, sign*2.*M_PI);
	angle -= sign*M_PI;
  }
  return angle;
}