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

#include <algorithm>
#include <stdexcept>
#include <svo/reprojector.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/map.h>
#include <svo/config.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <vikit/abstract_camera.h>
#include <vikit/math_utils.h>
#include <vikit/timer.h>

namespace svo {

Reprojector::Reprojector(vk::AbstractCamera* cam, Map& map) :
    map_(map)
{
  initializeGrid(cam);
}

Reprojector::~Reprojector()
{
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ delete c; });
}

void Reprojector::initializeGrid(vk::AbstractCamera* cam)
{
  grid_.cell_size = Config::gridSize();
  grid_.grid_n_cols = ceil(static_cast<double>(cam->width())/grid_.cell_size);
  grid_.grid_n_rows = ceil(static_cast<double>(cam->height())/grid_.cell_size);
  grid_.cells.resize(grid_.grid_n_cols*grid_.grid_n_rows);

  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell*& c){ 
    c = new Cell; 
  });

  grid_.cell_order.resize(grid_.cells.size());

  for(size_t i=0; i<grid_.cells.size(); ++i){
    grid_.cell_order[i] = i;
  }
  
  // maybe we should do it at every iteration!
  random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); 
}

void Reprojector::resetGrid()
{
  n_matches_ = 0;
  n_trials_ = 0;

  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell* c){ 
    c->clear(); 
  });
}

// 對鄰近頁框與地圖中可被當前頁框觀測到的點進行重投影
void Reprojector::reprojectMap(
    FramePtr frame,
    std::vector< std::pair<FramePtr,std::size_t> >& overlap_kfs)
{
  resetGrid();

  // Identify those Keyframes which share a common field of view.
  SVO_START_TIMER("reproject_kfs");
  list< pair<FramePtr,double> > close_kfs;

  // 取得當前頁框附近的頁框（兩頁框觀測到相同的關鍵點）
  map_.getCloseKeyframes(frame, close_kfs);

  // Sort KFs with overlap according to their closeness
  // 根據與當前頁框的距離排序，越近越前面
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  // Reproject all mappoints of the closest N kfs with overlap. We only store
  // in which grid cell the points fall.
  size_t n = 0;
  overlap_kfs.reserve(options_.max_n_kfs);

  // 遍歷與當前頁框相近的頁框們（只遍歷最接近的前 options_.max_n_kfs 個）
  for(auto it_frame=close_kfs.begin(), ite_frame=close_kfs.end();
      it_frame!=ite_frame && n < options_.max_n_kfs; ++it_frame, ++n)
  {
    FramePtr ref_frame = it_frame->first;
    overlap_kfs.push_back(pair<FramePtr,size_t>(ref_frame, 0));

    // 遍歷參考頁框的特徵點
    // Try to reproject each mappoint that the other KF observes
    for(auto it_ftr=ref_frame->fts_.begin(), ite_ftr=ref_frame->fts_.end();
        it_ftr!=ite_ftr; ++it_ftr)
    {
      // check if the feature has a mappoint assigned
      if((*it_ftr)->point == NULL){
        continue;
      }        

      // 避免當前頁框對相同的點進行重投影
      // make sure we project a point only once
      if((*it_ftr)->point->last_projected_kf_id_ == frame->id_){
        continue;
      }
      
      // 紀錄對該點進行除投影的頁框 id
      (*it_ftr)->point->last_projected_kf_id_ = frame->id_;

      // 檢查 point 是否在 frame 的成像平面上，若在就將該空間點與其投影位置封裝成 Candidate，由 Grid 來管理
      // (*it_ftr)->point 鄰近當前頁框的參考頁框上，特徵點觀測到的空間點
      if(reprojectPoint(frame, (*it_ftr)->point)){
        overlap_kfs.back().second++;
      }
    }
  }
  SVO_STOP_TIMER("reproject_kfs");

  // Now project all point candidates
  SVO_START_TIMER("reproject_candidates");
  {
    boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);

    // it 為 pair<Point*, Feature*> PointCandidate;
    // it->first 為 Point*
    auto it = map_.point_candidates_.candidates_.begin();

    while(it != map_.point_candidates_.candidates_.end())
    {
      // 檢查 point 是否在 frame 的成像平面上，若在就將該空間點與其投影位置封裝成 Candidate，由 Grid 來管理
      if(!reprojectPoint(frame, it->first))
      {
        it->first->n_failed_reproj_ += 3;

        // 若該空間點重投影失敗次數過多，表示該點可能是誤估計產生的
        if(it->first->n_failed_reproj_ > 30)
        {
          // 從地圖中移除
          map_.point_candidates_.deleteCandidate(*it);

          // 取得下一個 PointCandidate 的指標
          it = map_.point_candidates_.candidates_.erase(it);
          continue;
        }
      }
      ++it;
    }
  } // unlock the mutex when out of scope
  SVO_STOP_TIMER("reproject_candidates");

  // Now we go through each grid cell and select one point to match.
  // At the end, we should have at maximum one reprojected point per cell.
  SVO_START_TIMER("feature_align");

  for(size_t i=0; i<grid_.cells.size(); ++i)
  {
    // we prefer good quality points over unkown quality (more likely to match)
    // and unknown quality over candidates (position not optimized)
    if(reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame)){
      n_matches_++;
    }
      
    if(n_matches_ > (size_t) Config::maxFts()){
      break;
    }      
  }

  SVO_STOP_TIMER("feature_align");
}


// 衡量重投影點的品質
bool Reprojector::pointQualityComparator(Candidate& lhs, Candidate& rhs)
{
  if(lhs.pt->type_ > rhs.pt->type_){
    return true;
  }
    
  return false;
}

bool Reprojector::reprojectCell(Cell& cell, FramePtr frame)
{
  // 根據 cell 品質排序
  cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));
  Cell::iterator it=cell.begin();

  // 每個 Cell 包含多個 Candidate，即為 it
  while(it!=cell.end())
  {
    n_trials_++;

    // 若 cell 品質不佳
    if(it->pt->type_ == Point::TYPE_DELETED)
    {
      // 移除該 cell，並取得下一個 cell 的指標
      it = cell.erase(it);
      continue;
    }

    bool found_match = true;

    if(options_.find_match_direct){
      found_match = matcher_.findMatchDirect(*it->pt, *frame, it->px);
    }      

    if(!found_match)
    {
      it->pt->n_failed_reproj_++;
      
      if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15){
        map_.safeDeletePoint(it->pt);
      }
        
      if(it->pt->type_ == Point::TYPE_CANDIDATE  && it->pt->n_failed_reproj_ > 30){
        map_.point_candidates_.deleteCandidatePoint(it->pt);
      }
        
      it = cell.erase(it);
      continue;
    }

    it->pt->n_succeeded_reproj_++;

    if(it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10){
      it->pt->type_ = Point::TYPE_GOOD;
    }      

    Feature* new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
    frame->addFeature(new_feature);

    // Here we add a reference in the feature to the 3D point, the other way
    // round is only done if this frame is selected as keyframe.
    new_feature->point = it->pt;

    if(matcher_.ref_ftr_->type == Feature::EDGELET)
    {
      new_feature->type = Feature::EDGELET;
      new_feature->grad = matcher_.A_cur_ref_*matcher_.ref_ftr_->grad;
      new_feature->grad.normalize();
    }

    // If the keyframe is selected and we reproject the rest, we don't have to
    // check this point anymore.
    it = cell.erase(it);

    // Maximum one point per cell.
    return true;
  }
  return false;
}

// 檢查 point 是否在 frame 的成像平面上，若在就將該空間點與其投影位置封裝成 Candidate，由 Grid 來管理
bool Reprojector::reprojectPoint(FramePtr frame, Point* point)
{
  // point 由世界座標轉換到相機座標
  Vector2d px(frame->w2c(point->pos_));

  // 8px is the patch size in the matcher
  if(frame->cam_->isInFrame(px.cast<int>(), 8)) 
  {
    // 根據 px 所處區域，計算 CandidateGrid 的索引值
    const int k = static_cast<int>(px[1]/grid_.cell_size) * grid_.grid_n_cols
                + static_cast<int>(px[0]/grid_.cell_size);

    grid_.cells.at(k)->push_back(Candidate(point, px));

    return true;
  }

  return false;
}

} // namespace svo
