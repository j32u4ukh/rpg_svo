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

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

// 尋找第一幀的角點位置與方向，並返回初始化結果
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  // 清空前一幀的關鍵點，並釋放 Frame 的記憶體空間
  reset();

  // 尋找角點的位置與方向，分別存入 px_ref_ 和 f_ref_
  detectFeatures(frame_ref, px_ref_, f_ref_);

  // 要求第一張影像需要有大於 100 個角點，不然則應更換為紋理更為豐富的影像作為第一張
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }

  // Frame 的 shared_ptr
  frame_ref_ = frame_ref;

  // 將找到的角點位置 px_ref_ 存入 px_cur_
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());

  return SUCCESS;
}

// 根據第一二幀，找到兩幀之間相互對應的點，以及它們所對應的空間點。
// 利用深度的中位數來控制地圖規模，並使 Frame 和 Point* 管理著各自的 Feature
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  // ref_ 為前一幀的資訊
  // 追蹤第二幀當中的第一幀角點，若追蹤失敗，則將該角點移除
  // disparities_：前後幀同一角點的距離
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  // 檢查追蹤到的角點數量是否足夠
  if(disparities_.size() < Config::initMinTracked()){
    return FAILURE;
  }   

  // 取得第一二幀角點距離的中位數
  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");

  // 第一二幀之間需要有足夠的距離
  if(disparity < Config::initMinDisparity()){
    return NO_KEYFRAME;
  }
  
  // 利用單應性矩陣最小化重投影誤差，取得相機間位姿（舊往新）以及角點在空間中的位置
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);

  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

  // 檢查內點數量是否足夠
  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;

  // 取得追蹤到的點的 z 值（即深度）
  for(size_t i=0; i<xyz_in_cur_.size(); ++i){
    depth_vec.push_back((xyz_in_cur_[i]).z());
  }
  
  // 取得深度的中位數
  double scene_depth_median = vk::getMedian(depth_vec);

  // 利用深度的中位數來控制地圖規模
  double scale = Config::mapScale()/scene_depth_median;

  // world -> cur_frame = world -> ref_frame -> cur_frame
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;

  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();

  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    // cv::Point2f to Vector2d
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);

    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && 
    frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && 
    xyz_in_cur_[*it].z() > 0)
    {
      // 取得空間點的世界座標
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it] * scale);
      Point* new_point = new Point(pos);

      // 根據前面優化位姿後的點，形成 Feature 
      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));

      // 每個 Frame 有多個 Feature
      frame_cur->addFeature(ftr_cur);

      // Point* 管理著有觀察到自己的 Feature
      new_point->addFrameRef(ftr_cur);

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  
  return SUCCESS;
}

// 清空前一幀的關鍵點，並釋放 Frame 的記憶體空間
void KltHomographyInit::reset()
{
  // tracked keypoints in current frame.
  px_cur_.clear();

  // Frame 的 shared_ptr 手動釋放記憶體，須使用 reset()
  frame_ref_.reset();
}

void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());

  // 找到的特徵點存入 new_features
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  // px_vec：keypoints to be tracked in reference frame.
  // clear()：清空陣列; reserve()：分配指定長度的記憶體
  px_vec.clear(); 
  px_vec.reserve(new_features.size());

  f_vec.clear(); 
  f_vec.reserve(new_features.size());

  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    // 角點的位置
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));

    // 相機座標下的成像平面像素點（深度為焦距長）
    f_vec.push_back(ftr->f);

    delete ftr;
  });
}

// 追蹤第二幀當中的第一幀角點，若追蹤失敗，則將該角點移除
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);

  // 這裡的 px_ref 和 px_cur 為相同的內容，因為基於光流法對相同點位移不大的假設
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  // 將過光流法的計算後，若位移不大的點，應可被追蹤到

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();

  f_cur.clear(); 
  f_cur.reserve(px_cur.size());

  disparities.clear(); 
  disparities.reserve(px_cur.size());

  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    // 若光流法追蹤結果為失敗，則移除相對應的點（包含：前一幀角點的位置與方向、當前幀的位置）
    if(!status[i])
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }

    // 指向以相機為中心的單位球狀座標上的向量
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));

    // 計算前後兩幀同一角點的距離
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());

    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

// 利用單應性矩陣最小化重投影誤差，取得相機間位姿（舊往新）以及角點在空間中的位置
void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d > uv_ref(f_ref.size());
  vector<Vector2d > uv_cur(f_cur.size());

  for(size_t i = 0, i_max = f_ref.size(); i < i_max; ++i)
  {
    // vk::project2d 轉換為歸一化座標
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }

  // 單應性矩陣：描述兩頁框的運動 Homography.T_c2_from_c1
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();

  vector<int> outliers;

  // xyz_in_cur：排除離群點（outliers）的角點
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);

  // 位姿 from c1 to c2
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
