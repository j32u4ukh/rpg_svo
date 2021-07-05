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
#include <vikit/math_utils.h>
#include <svo/point.h>
#include <svo/frame.h>
#include <svo/feature.h>
 
namespace svo {

int Point::point_counter_ = 0;

Point::Point(const Vector3d& pos) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(0),
  v_pt_(NULL),
  last_published_ts_(0),
  last_projected_kf_id_(-1),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{}

Point::Point(const Vector3d& pos, Feature* ftr) :
  id_(point_counter_++),
  pos_(pos),
  normal_set_(false),
  n_obs_(1),
  v_pt_(NULL),
  last_published_ts_(0),
  last_projected_kf_id_(-1),
  type_(TYPE_UNKNOWN),
  n_failed_reproj_(0),
  n_succeeded_reproj_(0),
  last_structure_optim_(0)
{
  obs_.push_front(ftr);
}

Point::~Point()
{}

void Point::addFrameRef(Feature* ftr)
{
  obs_.push_front(ftr);
  ++n_obs_;
}

Feature* Point::findFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it){
    if((*it)->frame == frame){
      return *it;
    }
  }   
      
  return NULL;    // no keyframe found
}

bool Point::deleteFrameRef(Frame* frame)
{
  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    if((*it)->frame == frame)
    {
      obs_.erase(it);
      return true;
    }
  }
  return false;
}

void Point::initNormal()
{
  assert(!obs_.empty());
  const Feature* ftr = obs_.back();
  assert(ftr->frame != NULL);
  normal_ = ftr->frame->T_f_w_.rotation_matrix().transpose()*(-ftr->f);
  normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
  normal_set_ = true;
}

// 返回是否找到夾角最小的頁框，將 ftr 更新為『和當前頁框夾角最小的頁框（同樣有觀察到當前這個 Point）的特徵』
bool Point::getCloseViewObs(const Vector3d& framepos, Feature*& ftr) const
{
  // TODO: get frame with same point of view AND same pyramid level!
  // 當前幀到目前這個 Point 的距離
  Vector3d obs_dir(framepos - pos_);   
  obs_dir.normalize();

  // obs_：觀察到當前點的特徵們
  auto min_it=obs_.begin();
  double min_cos_angle = 0;

  for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
  {
    // (*it)->frame->pos()：取得該特徵所屬的 frame 的位置
    // 這裡產生遍歷所有觀測到這個 Point 的 frame，是否也包含當前這幀呢？
    // dir：當前點到該 frame 的向量
    Vector3d dir((*it)->frame->pos() - pos_); 
    dir.normalize();

    // 計算 obs_dir 和 dir 之間的 cos 值，由於兩向量長度皆為 1，cos 值相當於兩向量夾角的弧度
    double cos_angle = obs_dir.dot(dir);

    // TODO:這裡的條件是否寫反了？
    if(cos_angle > min_cos_angle)
    {
      min_cos_angle = cos_angle;
      min_it = it;
    }
  }

  // 更新 matcher 的 ref_ftr_
  ftr = *min_it;

  // 排除大於 60° 的情況
  // assume that observations larger than 60° are useless
  if(min_cos_angle < 0.5){
    return false;
  }
    
  return true;
}

void Point::optimize(const size_t n_iter)
{
  Vector3d old_point = pos_;
  double chi2 = 0.0;
  Matrix3d H;
  Vector3d b;

  for(size_t i=0; i<n_iter; i++)
  {
    H.setZero();
    b.setZero();
    double new_chi2 = 0.0;

    // compute residuals，遍歷每個 Feature*
    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
      Matrix23d J;

      // 當前這個點，轉換到含有『特徵 it』的相機座標系下
      const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_);
      Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_.rotation_matrix(), J);

      // 像素座標誤差 e：成像平面上的 (u, v) 和 實際測量值 相減
      const Vector2d e(vk::project2d((*it)->f) - vk::project2d(p_in_f));
      new_chi2 += e.squaredNorm();

      /* noalias()
      在 Eigen 中，當變量同時出現在左值和右值，賦值操作可能會帶來混淆問題。一般的操作，Eigen 默認都是存在混淆的。
      所以 Eigen 對矩陣乘法自動引入了臨時變量，對的 matA = matA * matA 這是必須的，
      但是對 matB = matA * matA 這樣便是不必要的了。我們可以使用 noalias() 函數來聲明這里沒有混淆，
      matA * matA 的結果可以直接賦值為 matB。matB.noalias() = matA * matA;*/
      H.noalias() += J.transpose() * J;
      b.noalias() -= J.transpose() * e;
    }

    // solve linear system H * dp = b
    const Vector3d dp(H.ldlt().solve(b));

    // check if error increased(第一次不納入)
    if((i > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dp[0]))
    {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << i
           << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
#endif
      pos_ = old_point; // roll-back
      break;
    }

    // update the model
    Vector3d new_point = pos_ + dp;
    old_point = pos_;
    pos_ = new_point;

    // 更新 chi2，將被用於檢查誤差是否變大
    chi2 = new_chi2;

#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << i
         << "\t Success \t new_chi2 = " << new_chi2
         << "\t norm(b) = " << vk::norm_max(b)
         << endl;
#endif

    // 若更新幅度極小，提前結束優化
    // stop when converged
    if(vk::norm_max(dp) <= EPS){
      break;
    }
      
  }
#ifdef POINT_OPTIMIZER_DEBUG
  cout << endl;
#endif
}

} // namespace svo
