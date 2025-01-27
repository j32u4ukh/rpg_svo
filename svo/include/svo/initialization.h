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

#ifndef SVO_INITIALIZATION_H
#define SVO_INITIALIZATION_H

#include <svo/global.h>

namespace svo {

class FrameHandlerMono;

/// Bootstrapping the map from the first two views.
namespace initialization {

enum InitResult { FAILURE, NO_KEYFRAME, SUCCESS };

/// Tracks features using Lucas-Kanade tracker and then estimates a homography.
/* Homography 單應矩陣
若場景中的特徵點都落在同一平面上(如牆、地面等)，則可透過單應性進行運動估計，
單應矩陣(Homography) H 描述了兩個平面之間的對應關係。
專案 svo 的目標是重建無人機的俯視相機的影像，因此這裡使用 Homography 來追蹤特徵點
*/ 
class KltHomographyInit {
  /* friend class
  在定義類別成員時，私用成員只能被同一個類別定義的成員存取，不可以直接由外界進行存取，
  然而有些時候，您希望提供私用成員給某些外部函式來存取，這時您 可以設定類別的「好友」，只有好友才可以直接存取自家的私用成員。
  參考：https://openhome.cc/Gossip/CppGossip/friendFunctionClass.html
  */
  friend class svo::FrameHandlerMono;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Frame 的 shared_ptr
  FramePtr frame_ref_;

  KltHomographyInit() {};
  ~KltHomographyInit() {};
  InitResult addFirstFrame(FramePtr frame_ref);
  InitResult addSecondFrame(FramePtr frame_ref);
  void reset();

protected:
  // keypoints to be tracked in reference frame.
  vector<cv::Point2f> px_ref_;

  // tracked keypoints in current frame.
  vector<cv::Point2f> px_cur_;
  
  // bearing vectors corresponding to the keypoints in the reference image.
  vector<Vector3d> f_ref_;
  
  // bearing vectors corresponding to the keypoints in the current image.
  vector<Vector3d> f_cur_;
  
  // 前後幀各個角點之間的距離 disparity between first and second frame.
  vector<double> disparities_;      

  vector<int> inliers_;             //!< inliers after the geometric check (e.g., Homography).
  vector<Vector3d> xyz_in_cur_;     //!< 3D points computed during the geometric check.
  SE3 T_cur_from_ref_;              //!< computed transformation between the first two frames.
};

/// Detect Fast corners in the image.
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec);

/// Compute optical flow (Lucas Kanade) for selected keypoints.
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities);

void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref);

} // namespace initialization
} // namespace svo

#endif // SVO_INITIALIZATION_H
