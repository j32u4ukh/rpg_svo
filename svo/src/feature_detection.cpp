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

#include <svo/feature_detection.h>
#include <svo/feature.h>
#include <fast/fast.h>
#include <vikit/vision.h>

namespace svo {
namespace feature_detection {

AbstractDetector::AbstractDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        cell_size_(cell_size),
        n_pyr_levels_(n_pyr_levels),
        grid_n_cols_(ceil(static_cast<double>(img_width)/cell_size_)),
        grid_n_rows_(ceil(static_cast<double>(img_height)/cell_size_)),
        grid_occupancy_(grid_n_cols_*grid_n_rows_, false)
{}

void AbstractDetector::resetGrid()
{
  std::fill(grid_occupancy_.begin(), grid_occupancy_.end(), false);
}

void AbstractDetector::setExistingFeatures(const Features& fts)
{
  std::for_each(fts.begin(), fts.end(), [&](Feature* i){
    grid_occupancy_.at(
        static_cast<int>(i->px[1]/cell_size_)*grid_n_cols_
        + static_cast<int>(i->px[0]/cell_size_)) = true;
  });
}

// 將像素點所在網格設置為被佔據
void AbstractDetector::setGridOccpuancy(const Vector2d& px)
{
  grid_occupancy_.at(
      static_cast<int>(px[1]/cell_size_)*grid_n_cols_
    + static_cast<int>(px[0]/cell_size_)) = true;
}

FastDetector::FastDetector(
    const int img_width,
    const int img_height,
    const int cell_size,
    const int n_pyr_levels) :
        AbstractDetector(img_width, img_height, cell_size, n_pyr_levels)
{}

void FastDetector::detect(
    Frame* frame,
    const ImgPyr& img_pyr,
    const double detection_threshold,
    Features& fts)
{
  // typedef vector<Corner> Corners;
  // Corner(int x, int y, float score, int level, float angle)
  // vector 初始化：長度為 grid_n_cols_ * grid_n_rows_，預設值為 Corner(0, 0, detection_threshold, 0, 0.0f)
  Corners corners(grid_n_cols_ * grid_n_rows_, Corner(0, 0, detection_threshold, 0, 0.0f));

  // 影像金字塔，根據不同『金字塔層級 L』來取得不同尺度下的特徵
  for(int L = 0; L < n_pyr_levels_; ++L)
  {
    // 縮放尺度 << 位元往左 1 位，相當於乘以 2
    const int scale = (1 << L);

    // 存放找到的角點
    vector<fast::fast_xy> fast_corners;

#if __SSE2__
      fast::fast_corner_detect_10_sse2(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#elif HAVE_FAST_NEON
      fast::fast_corner_detect_9_neon(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#else
      fast::fast_corner_detect_10(
          (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols,
          img_pyr[L].rows, img_pyr[L].cols, 20, fast_corners);
#endif

    // 針對找到的角點進行評價，並返回高於門檻值的角點
    vector<int> scores, nm_corners;
    fast::fast_corner_score_10(
      (fast::fast_byte*) img_pyr[L].data, img_pyr[L].cols, fast_corners, 20, scores);
    fast::fast_nonmax_3x3(fast_corners, scores, nm_corners);

    for(auto it=nm_corners.begin(), ite=nm_corners.end(); it!=ite; ++it)
    {
      // 依序取出分數高的角點
      fast::fast_xy& xy = fast_corners.at(*it);

      // 根據縮放尺度(scale)計算網格中的索引值
      const int k = static_cast<int>((xy.y*scale)/cell_size_)*grid_n_cols_
                  + static_cast<int>((xy.x*scale)/cell_size_);

      if(grid_occupancy_[k]){
        continue;
      }
        
      const float score = vk::shiTomasiScore(img_pyr[L], xy.x, xy.y);
      
      // 若當前分數較原先網格內的高，則取代它成為新的數據
      if(score > corners.at(k).score){
        corners.at(k) = Corner(xy.x*scale, xy.y*scale, score, L, 0.0f);
      }        
    }
  }

  // Create feature for every corner that has high enough corner score
  std::for_each(corners.begin(), corners.end(), [&](Corner& c) {
    // 若角點的分數 高於 偵測用門檻
    if(c.score > detection_threshold){

      // 形成特徵點
      fts.push_back(new Feature(frame, Vector2d(c.x, c.y), c.level));
    }      
  });

  resetGrid();
}

} // namespace feature_detection
} // namespace svo

