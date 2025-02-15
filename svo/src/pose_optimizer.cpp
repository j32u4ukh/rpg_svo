// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <stdexcept>
#include <svo/pose_optimizer.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <vikit/robust_cost.h>
#include <vikit/math_utils.h>

namespace svo {
namespace pose_optimizer {

void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs)
{
  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  vk::robust_cost::TukeyWeightFunction weight_function;

  // 世界座標 轉 相機座標
  SE3 T_old(frame->T_f_w_);

  Matrix6d H;
  Vector6d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors; 
  errors.reserve(frame->fts_.size());

  // 遍歷每個 Feature*
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL){
      continue;
    }
    
    // 像素座標誤差 e：成像平面上的 (u, v) 和 實際測量值 相減
    // 是 3D 點的投影位置和觀測位置的差，因此又稱『重投影誤差』
    Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->point->pos_);

    // 根據影像金字塔的 level，調整誤差的尺度
    e *= 1.0 / (1<<(*it)->level);

    errors.push_back(e.norm());
  }

  // 當全部 (*it)->point 都是 NULL 才會發生
  if(errors.empty()){
    return;
  }
    
  vk::robust_cost::MADScaleEstimator scale_estimator;
  estimated_scale = scale_estimator.compute(errors);

  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale = estimated_scale;

  for(size_t iter=0; iter<n_iter; iter++)
  {
    // overwrite scale
    if(iter == 5){
      // 若誤差小於 0.85，則 scale 會大於 1; 反之則會小於 1
      scale = 0.85/frame->cam_->errorMultiplier2();
    }
    
    b.setZero();
    H.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point == NULL){
        continue;
      }
        
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);

      // 求取雅可比矩陣 J(2 X 6)
      Frame::jacobian_xyz2uv(xyz_f, J);

      // 像素座標誤差 e：成像平面上的 (u, v) 和 實際測量值 相減
      // 是 3D 點的投影位置和觀測位置的差，因此又稱『重投影誤差』
      Vector2d e = vk::project2d((*it)->f) - vk::project2d(xyz_f);

      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);

      // 根據影像金字塔的 level，調整誤差的尺度
      e *= sqrt_inv_cov;

      if(iter == 0){
        // just for debug
        chi2_vec_init.push_back(e.squaredNorm()); 
      }
      
      // 根據影像金字塔的 level，調整雅可比矩陣 J 的尺度
      J *= sqrt_inv_cov;

      double weight = weight_function.value(e.norm()/scale);

      H.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.squaredNorm()*weight;
    }

    // solve linear system H * dT = b
    const Vector6d dT(H.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose){
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      }
      
      // roll-back
      frame->T_f_w_ = T_old; 

      break;
    }

    // update the model 左乘擾動 SE3::exp(dT) 李代數轉李群的形式
    SE3 T_new = SE3::exp(dT) * frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;

    if(verbose){
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << vk::norm_max(dT) << std::endl;
    }

    // stop when converged
    if(vk::norm_max(dT) <= EPS){
      break;
    }      
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  const double pixel_variance=1.0;
  frame->Cov_ = pixel_variance*(H*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
  size_t n_deleted_refs = 0;

  for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL){
      continue;
    }
      
    Vector2d e = vk::project2d((*it)->f) - vk::project2d(frame->T_f_w_ * (*it)->point->pos_);
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());

    if(e.norm() > reproj_thresh_scaled)
    {
      // we don't need to delete a reference in the point since it was not created yet
      (*it)->point = NULL;
      ++n_deleted_refs;
    }
  }

  error_init=0.0;
  error_final=0.0;

  if(!chi2_vec_init.empty()){
    error_init = sqrt(vk::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  }
    
  if(!chi2_vec_final.empty()){
    error_final = sqrt(vk::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();
  }
  
  estimated_scale *= frame->cam_->errorMultiplier2();

  if(verbose){
    std::cout << "n deleted obs = " << n_deleted_refs
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  }
    
  num_obs -= n_deleted_refs;
}

} // namespace pose_optimizer
} // namespace svo
