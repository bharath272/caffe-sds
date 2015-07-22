// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <cfloat>

#include "caffe/sds_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void CFMPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
}

template <typename Dtype>
void CFMPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  num_levels_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  cfm_weights_.Reshape(bottom[1]->num(), 1, height_,
      width_);
  if(top.size()>1){
    top[1]->Reshape(bottom[1]->num(), 1, height_,
      width_);

  }
  mask_width_ = bottom[2]->width();
  mask_height_ = bottom[2]->height();
}

template <typename Dtype>
void CFMPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);


  Dtype* cfm_data = cfm_weights_.mutable_cpu_data();
  const Dtype* mask_data = bottom[2]->cpu_data();
  //First set the cfm maps
  caffe_set(cfm_weights_.count(), Dtype(0.0), cfm_data);
  for (int n=0; n<num_rois; ++n) {
    int box_xmin = bottom_rois[5*n+1]; 
    int box_ymin = bottom_rois[5*n+2];
    int box_xmax = bottom_rois[5*n+3];
    int box_ymax = bottom_rois[5*n+4];
    int box_w = box_xmax-box_xmin+1.0;
    int box_h = box_ymax-box_ymin+1.0;    
    Dtype cell_w = static_cast<Dtype>(mask_width_)/static_cast<Dtype>(box_w); 
    Dtype cell_h = static_cast<Dtype>(mask_height_)/static_cast<Dtype>(box_h);
    Dtype* cfm_data_this = cfm_data + cfm_weights_.offset(n);
    const Dtype* mask_data_this = mask_data + bottom[2]->offset(n);
    for (int y=box_ymin; y<=box_ymax; y++) {
      for (int x=box_xmin; x<=box_xmax; x++) {
        //compute where the box falls in the mask
        Dtype Ycenter = static_cast<Dtype>(y+0.5-box_ymin)*cell_h-0.5;
        int Ymin = static_cast<int>(ceil(Ycenter-cell_h));
        int Ymax = static_cast<int>(floor(Ycenter+cell_h));  
        Dtype Xcenter = static_cast<Dtype>(x+0.5-box_xmin)*cell_w-0.5;
        int Xmin = static_cast<int>(ceil(Xcenter-cell_w));
        int Xmax = static_cast<int>(floor(Xcenter+cell_w));  

        Ymin = max(Ymin,0);
        Xmin = max(Xmin,0);
        Ymax = min(Ymax, mask_height_-1);
        Xmax = min(Xmax, mask_width_-1);
        Dtype val = 0;
        Dtype count = 0;
        for(int Y=Ymin; Y<=Ymax; Y++){
          for(int X=Xmin; X<=Xmax; X++){
            Dtype wty = Dtype(1.0) - fabs(static_cast<Dtype>(Y)-Ycenter)/cell_h;
            Dtype wtx = Dtype(1.0) - fabs(static_cast<Dtype>(X)-Xcenter)/cell_w;
            val += mask_data_this[Y*mask_width_ + X]*wty*wtx;
            count +=wty*wtx;
          }
        }
        count=count==0?1:count;
        cfm_data_this[y*width_+x] = static_cast<Dtype>(val/count>=0.5);
      }
    }

  }
  // For each ROI R = [level x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_level = bottom_rois[0];
    int roi_start_w = bottom_rois[1];
    int roi_start_h = bottom_rois[2];
    int roi_end_w = bottom_rois[3];
    int roi_end_h = bottom_rois[4];
    CHECK_GE(roi_level, 0);
    CHECK_LT(roi_level, num_levels_);
    CHECK_GE(roi_start_w, 0);
    CHECK_GE(roi_start_h, 0);
    CHECK_LT(roi_end_w, width_);
    CHECK_LT(roi_end_h, height_);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(pooled_width_);

    const Dtype* level_data = bottom_data + bottom[0]->offset(roi_level);
    Dtype* cfm_data_box = cfm_data + cfm_weights_.offset(n);
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);

          CHECK_GT(hend, hstart);
          CHECK_GT(wend, wstart);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              Dtype val = level_data[index]*cfm_data_box[index];
              //LOG(INFO)<<val<<" "<<cfm_data_box[index];
              if (val > top_data[pool_index]) {
                top_data[pool_index] = val;
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel
      level_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
  if(top.size()>1){
    caffe_copy(cfm_weights_.count(),cfm_weights_.cpu_data(), top[1]->mutable_cpu_data());
  }
}

template <typename Dtype>
void CFMPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(CFMPoolingLayer);
#endif

INSTANTIATE_CLASS(CFMPoolingLayer);
REGISTER_LAYER_CLASS(CFMPooling);

}  // namespace caffe
