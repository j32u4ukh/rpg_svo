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

#include <set>
#include <svo/map.h>
#include <svo/point.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <boost/bind.hpp>

namespace svo {

Map::Map() {}

Map::~Map()
{
  reset();
  SVO_INFO_STREAM("Map destructed");
}

void Map::reset()
{
  keyframes_.clear();
  point_candidates_.reset();
  emptyTrash();
}

bool Map::safeDeleteFrame(FramePtr frame)
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    if(*it == frame)
    {
      std::for_each((*it)->fts_.begin(), (*it)->fts_.end(), [&](Feature* ftr){
        removePtFrameRef(it->get(), ftr);
      });
      keyframes_.erase(it);
      found = true;
      break;
    }
  }

  point_candidates_.removeFrameCandidates(frame);

  if(found)
    return true;

  SVO_ERROR_STREAM("Tried to delete Keyframe in map which was not there.");
  return false;
}

void Map::removePtFrameRef(Frame* frame, Feature* ftr)
{
  if(ftr->point == NULL)
    return; // mappoint may have been deleted in a previous ref. removal
  Point* pt = ftr->point;
  ftr->point = NULL;
  if(pt->obs_.size() <= 2)
  {
    // If the references list of mappoint has only size=2, delete mappoint
    safeDeletePoint(pt);
    return;
  }
  pt->deleteFrameRef(frame);  // Remove reference from map_point
  frame->removeKeyPoint(ftr); // Check if mp was keyMp in keyframe
}

void Map::safeDeletePoint(Point* pt)
{
  // Delete references to mappoints in all keyframes
  // 遍歷觀察到這個點 pt 的特徵點
  std::for_each(pt->obs_.begin(), pt->obs_.end(), [&](Feature* ftr){
    ftr->point=NULL;
    ftr->frame->removeKeyPoint(ftr);
  });

  pt->obs_.clear();

  // Delete mappoint
  // 將該 pt 標注為 Point::TYPE_DELETED，並利用 trash_points_ 進行管理，在適當的時機再進行刪除
  deletePoint(pt);
}

// 將該 pt 標注為 Point::TYPE_DELETED，並利用 trash_points_ 進行管理，在適當的時機再進行刪除
void Map::deletePoint(Point* pt)
{
  pt->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(pt);
}

void Map::addKeyframe(FramePtr new_keyframe)
{
  keyframes_.push_back(new_keyframe);
}

void Map::getCloseKeyframes(
    const FramePtr& frame,
    std::list< std::pair<FramePtr,double> >& close_kfs) const
{
  for(auto kf : keyframes_)
  {
    // check if kf has overlaping field of view with frame, use therefore KeyPoints
    for(auto keypoint : kf->key_pts_)
    {
      if(keypoint == nullptr){
        continue;
      }

      // 將點 xyz_w 轉換到相機座標系下，進而判斷相機能否看到該點（是否在相機的前面，且在成像平面的投影範圍內）
      // frame 可以看到 kf 的關鍵點，表示兩者觀測著差不多的區域，兩幀之間的距離也應相近
      if(frame->isVisible(keypoint->point->pos_))
      {
        // 和當前 Frame 觀測到相同的關鍵點的關鍵幀，以及和它的距離
        close_kfs.push_back(
            std::make_pair(kf, (frame->T_f_w_.translation() - kf->T_f_w_.translation()).norm()));

        // this keyframe has an overlapping field of view -> add to close_kfs
        break; 
      }
    }
  }
}

FramePtr Map::getClosestKeyframe(const FramePtr& frame) const
{
  list< pair<FramePtr,double> > close_kfs;
  getCloseKeyframes(frame, close_kfs);
  if(close_kfs.empty())
  {
    return nullptr;
  }


  // Sort KFs with overlap according to their closeness
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
                 boost::bind(&std::pair<FramePtr, double>::second, _2));

  if(close_kfs.front().first != frame)
    return close_kfs.front().first;
  close_kfs.pop_front();
  return close_kfs.front().first;
}

FramePtr Map::getFurthestKeyframe(const Vector3d& pos) const
{
  FramePtr furthest_kf;
  double maxdist = 0.0;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    double dist = ((*it)->pos()-pos).norm();
    if(dist > maxdist) {
      maxdist = dist;
      furthest_kf = *it;
    }
  }
  return furthest_kf;
}

bool Map::getKeyframeById(const int id, FramePtr& frame) const
{
  bool found = false;
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
    if((*it)->id_ == id) {
      found = true;
      frame = *it;
      break;
    }
  return found;
}

void Map::transform(const Matrix3d& R, const Vector3d& t, const double& s)
{
  for(auto it=keyframes_.begin(), ite=keyframes_.end(); it!=ite; ++it)
  {
    Vector3d pos = s*R*(*it)->pos() + t;
    Matrix3d rot = R*(*it)->T_f_w_.rotation_matrix().inverse();
    (*it)->T_f_w_ = SE3(rot, pos).inverse();
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      if((*ftr)->point->last_published_ts_ == -1000)
        continue;
      (*ftr)->point->last_published_ts_ = -1000;
      (*ftr)->point->pos_ = s*R*(*ftr)->point->pos_ + t;
    }
  }
}

void Map::emptyTrash()
{
  /*
  std::for_each
  參考：https://blog.csdn.net/u014613043/article/details/50619254

  *& 指標的引用
  void*& func(Point*& pt) 傳入的是 Point*，pt 是該指標的引用，在函式內對 pt 做修改，傳入的外部 Point* 也會被修改
  參考：https://www.zhihu.com/question/19977090

  lambda
  [=]：lambda-introducer，也稱為 capture clause。
  所有的 lambda expression 都是以它來作為開頭，不可以省略，它除了用來作為 lambda expression 開頭的關鍵字之外，
  也有抓取（capture）變數的功能，指定該如何將目前 scope 範圍之變數抓取至 lambda expression 中使用，而抓取變數的方式
  則分為傳值（by value）與傳參考（by reference）兩種，跟一般函數參數的傳入方式類似，不過其語法有些不同，以下我們以
  範例解釋：

        []：只有兩個中括號，完全不抓取外部的變數。
        [=]：所有的變數都以傳值（by value）的方式抓取。
        [&]：所有的變數都以傳參考（by reference）的方式抓取。
        [x, &y]：x 變數使用傳值、y 變數使用傳參考。
        [=, &y]：除了 y 變數使用傳參考之外，其餘的變數皆使用傳值的方式。
        [&, x]：除了 x 變數使用傳值之外，其餘的變數皆使用傳參考的方式。

  這裡要注意一點，預設的抓取選項（capture-default，亦即 = 或是 &）要放在所有的項目之前，也就是放在第一個位置。
  參考：https://blog.gtwang.org/programming/lambda-expression-in-c11/
  */
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& pt){
    // 釋放 pt 所指向的記憶體區域
    delete pt;

    // 指標指向 NULL
    pt = NULL;

    // 參考：https://stackoverflow.com/questions/13223399/deleting-a-pointer-in-c
  });

  /*
  list.clear()
  清空 std::list 容器
  參考：https://www.cplusplus.com/reference/list/list/clear/
  */
  trash_points_.clear();
  point_candidates_.emptyTrash();
}

MapPointCandidates::MapPointCandidates(){}

MapPointCandidates::~MapPointCandidates()
{
  reset();
}

void MapPointCandidates::newCandidatePoint(Point* point, double depth_sigma2)
{
  point->type_ = Point::TYPE_CANDIDATE;
  boost::unique_lock<boost::mutex> lock(mut_);
  candidates_.push_back(PointCandidate(point, point->obs_.front()));
}

// 將候選點產生的特徵點，加入該特徵點的頁框的特徵點（it->second->frame->addFeature(it->second)）
void MapPointCandidates::addCandidatePointToFrame(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  PointCandidateList::iterator it=candidates_.begin();

  // it typedef pair<Point*, Feature*> PointCandidate;
  while(it != candidates_.end())
  {
    // it->first Point*
    if(it->first->obs_.front()->frame == frame.get())
    {
      // insert feature in the frame
      it->first->type_ = Point::TYPE_UNKNOWN;
      it->first->n_failed_reproj_ = 0;

      // it->second Feature*
      it->second->frame->addFeature(it->second);

      it = candidates_.erase(it);
    }
    else{
      ++it;
    }      
  }
}

bool MapPointCandidates::deleteCandidatePoint(Point* point)
{
  boost::unique_lock<boost::mutex> lock(mut_);

  // PointCandidateList candidates_ 包含多個 PointCandidate
  // 又 typedef pair<Point*, Feature*> PointCandidate;
  for(auto it=candidates_.begin(), ite=candidates_.end(); it!=ite; ++it)
  {
    // it->first 為 Point*
    if(it->first == point)
    {
      deleteCandidate(*it);

      /* list 中 remove 和 erase 都是刪除一個元素，其中 remove 引數型別和資料型別一致，而 erase 引數型別是迭代器。
      remove（aim）是刪除連結串列中的 aim 元素，若有多個 aim，都會刪除，而
      erase（it）是刪除迭代器指定位置的元素，並且返回下一個位置的迭代器。
      erase 還能透過指定刪除迭代器的起點和終點，一次刪除多筆數據。

      參考：
      https://www.itread01.com/content/1556144404.html
      https://www.cplusplus.com/reference/list/list/erase/
      */
      candidates_.erase(it);
      return true;
    }
  }

  return false;
}

void MapPointCandidates::removeFrameCandidates(FramePtr frame)
{
  boost::unique_lock<boost::mutex> lock(mut_);
  auto it=candidates_.begin();
  while(it!=candidates_.end())
  {
    if(it->second->frame == frame.get())
    {
      deleteCandidate(*it);
      it = candidates_.erase(it);
    }
    else
      ++it;
  }
}

void MapPointCandidates::reset()
{
  boost::unique_lock<boost::mutex> lock(mut_);
  std::for_each(candidates_.begin(), candidates_.end(), [&](PointCandidate& c){
    delete c.first;
    delete c.second;
  });
  candidates_.clear();
}

void MapPointCandidates::deleteCandidate(PointCandidate& c)
{
  // 其他頁框可能仍指向這個 PointCandidate，因此上不能直接將它刪除，而是對它進行標注要刪除
  // camera-rig: another frame might still be pointing to the candidate point
  // therefore, we can't delete it right now.
  delete c.second; 
  c.second=NULL;
  c.first->type_ = Point::TYPE_DELETED;
  trash_points_.push_back(c.first);
}

void MapPointCandidates::emptyTrash()
{
  std::for_each(trash_points_.begin(), trash_points_.end(), [&](Point*& p){
    delete p; 
    p = NULL;
  });

  trash_points_.clear();
}

namespace map_debug {

void mapValidation(Map* map, int id)
{
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    frameValidation(it->get(), id);
}

void frameValidation(Frame* frame, int id)
{
  for(auto it = frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point==NULL)
      continue;

    if((*it)->point->type_ == Point::TYPE_DELETED)
      printf("ERROR DataValidation %i: Referenced point was deleted.\n", id);

    if(!(*it)->point->findFrameRef(frame))
      printf("ERROR DataValidation %i: Frame has reference but point does not have a reference back.\n", id);

    pointValidation((*it)->point, id);
  }
  for(auto it=frame->key_pts_.begin(); it!=frame->key_pts_.end(); ++it)
    if(*it != NULL)
      if((*it)->point == NULL)
        printf("ERROR DataValidation %i: KeyPoints not correct!\n", id);
}

void pointValidation(Point* point, int id)
{
  for(auto it=point->obs_.begin(); it!=point->obs_.end(); ++it)
  {
    bool found=false;
    for(auto it_ftr=(*it)->frame->fts_.begin(); it_ftr!=(*it)->frame->fts_.end(); ++it_ftr)
     if((*it_ftr)->point == point) {
       found=true; break;
     }
    if(!found)
      printf("ERROR DataValidation %i: Point %i has inconsistent reference in frame %i, is candidate = %i\n", id, point->id_, (*it)->frame->id_, (int) point->type_);
  }
}

void mapStatistics(Map* map)
{
  // compute average number of features which each frame observes
  size_t n_pt_obs(0);
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
    n_pt_obs += (*it)->nObs();
  printf("\n\nMap Statistics: Frame avg. point obs = %f\n", (float) n_pt_obs/map->size());

  // compute average number of observations that each point has
  size_t n_frame_obs(0);
  size_t n_pts(0);
  std::set<Point*> points;
  for(auto it=map->keyframes_.begin(); it!=map->keyframes_.end(); ++it)
  {
    for(auto ftr=(*it)->fts_.begin(); ftr!=(*it)->fts_.end(); ++ftr)
    {
      if((*ftr)->point == NULL)
        continue;
      if(points.insert((*ftr)->point).second) {
        ++n_pts;
        n_frame_obs += (*ftr)->point->nRefs();
      }
    }
  }
  printf("Map Statistics: Point avg. frame obs = %f\n\n", (float) n_frame_obs/n_pts);
}

} // namespace map_debug
} // namespace svo
