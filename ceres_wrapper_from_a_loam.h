// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>

struct LidarEdgeFactor
{
	LidarEdgeFactor(const Eigen::Vector3d &curr_point,
                  const Eigen::Vector3d &last_point_a,
                  const Eigen::Vector3d &last_point_b,
                  double s)
		: curr_point_(curr_point),
      last_point_a_(last_point_a),
      last_point_b_(last_point_b), s_(s) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point_.x()), T(curr_point_.y()), T(curr_point_.z())};
		Eigen::Matrix<T, 3, 1> lpa{T(last_point_a_.x()), T(last_point_a_.y()), T(last_point_a_.z())};
		Eigen::Matrix<T, 3, 1> lpb{T(last_point_b_.x()), T(last_point_b_.y()), T(last_point_b_.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s_), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s_) * t[0], T(s_) * t[1], T(s_) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
		Eigen::Matrix<T, 3, 1> de = lpa - lpb;

		residual[0] = nu.x() / de.norm();
		residual[1] = nu.y() / de.norm();
		residual[2] = nu.z() / de.norm();

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                     const Eigen::Vector3d &last_point_a,
                                     const Eigen::Vector3d &last_point_b,
                                     double s)
	{
		return new ceres::AutoDiffCostFunction<LidarEdgeFactor, 3, 4, 3>(new LidarEdgeFactor(curr_point, last_point_a, last_point_b, s));
	}

	Eigen::Vector3d curr_point_, last_point_a_, last_point_b_;
	double s_;
};

struct LidarPlaneFactor
{
	LidarPlaneFactor(const Eigen::Vector3d &curr_point,
                   const Eigen::Vector3d &last_point_j,
                   const Eigen::Vector3d &last_point_l,
                   const Eigen::Vector3d &last_point_m,
                   double s)
		: curr_point_(curr_point),
      last_point_j_(last_point_j),
      last_point_l_(last_point_l),
      last_point_m_(last_point_m), s_(s)
	{
		ljm_norm = (last_point_j_ - last_point_l_).cross(last_point_j_ - last_point_m_);
		ljm_norm.normalize();
	}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{

		Eigen::Matrix<T, 3, 1> cp{T(curr_point_.x()), T(curr_point_.y()), T(curr_point_.z())};
		Eigen::Matrix<T, 3, 1> lpj{T(last_point_j_.x()), T(last_point_j_.y()), T(last_point_j_.z())};
		Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

		Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
		Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
		q_last_curr = q_identity.slerp(T(s_), q_last_curr);
		Eigen::Matrix<T, 3, 1> t_last_curr{T(s_) * t[0], T(s_) * t[1], T(s_) * t[2]};

		Eigen::Matrix<T, 3, 1> lp;
		lp = q_last_curr * cp + t_last_curr;

		residual[0] = (lp - lpj).dot(ljm);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                    const Eigen::Vector3d &last_point_j,
                                     const Eigen::Vector3d &last_point_l,
                                     const Eigen::Vector3d &last_point_m,
                                     const double s)
	{
		return new ceres::AutoDiffCostFunction<LidarPlaneFactor, 1, 4, 3>(new LidarPlaneFactor(curr_point, last_point_j, last_point_l, last_point_m, s));
	}

	Eigen::Vector3d curr_point_, last_point_j_, last_point_l_, last_point_m_;
	Eigen::Vector3d ljm_norm;
	double s_;
};

struct LidarPlaneNormFactor
{
	LidarPlaneNormFactor(const Eigen::Vector3d &curr_point,
                       const Eigen::Vector3d &plane_unit_norm,
                       double negative_OA_dot_norm)
  : curr_point_(curr_point),
    plane_unit_norm_(plane_unit_norm),
    negative_OA_dot_norm_(negative_OA_dot_norm) {}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point_.x()), T(curr_point_.y()), T(curr_point_.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;

		Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm_.x()), T(plane_unit_norm_.y()), T(plane_unit_norm_.z()));
		residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm_);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point, const Eigen::Vector3d &plane_unit_norm,
									   double negative_OA_dot_norm_)
	{
		return new ceres::AutoDiffCostFunction<LidarPlaneNormFactor, 1, 4, 3>(new LidarPlaneNormFactor(curr_point, plane_unit_norm, negative_OA_dot_norm_));
	}

	Eigen::Vector3d curr_point_;
	Eigen::Vector3d plane_unit_norm_;
	double negative_OA_dot_norm_;
};


struct LidarDistanceFactor
{
	LidarDistanceFactor(const Eigen::Vector3d &curr_point,
                      const Eigen::Vector3d &closed_point) 
						: curr_point_(curr_point), closed_point_(closed_point){}

	template <typename T>
	bool operator()(const T *q, const T *t, T *residual) const
	{
		Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
		Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
		Eigen::Matrix<T, 3, 1> cp{T(curr_point_.x()), T(curr_point_.y()), T(curr_point_.z())};
		Eigen::Matrix<T, 3, 1> point_w;
		point_w = q_w_curr * cp + t_w_curr;


		residual[0] = point_w.x() - T(closed_point_.x());
		residual[1] = point_w.y() - T(closed_point_.y());
		residual[2] = point_w.z() - T(closed_point_.z());
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point,
                                      const Eigen::Vector3d &closed_point)
	{
		return new ceres::AutoDiffCostFunction<LidarDistanceFactor, 3, 4, 3>(new LidarDistanceFactor(curr_point, closed_point));
	}

	Eigen::Vector3d curr_point_;
	Eigen::Vector3d closed_point_;
};
