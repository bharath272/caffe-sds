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

namespace caffe {

template <typename Dtype>
__global__ void GetCFMWeights(const int nthreads, const Dtype* mask_data,
  const Dtype* bottom_rois, Dtype* cfm_data, const int mask_width_, const int mask_height_,
  const int width_, const int height_){
  CUDA_KERNEL_LOOP(index, nthreads) {

  //get where in the pooling map we are
  int x = index % width_;
  int y = (index / width_) % height_;
  int n = (index / width_) / height_;
  
  //get the box coordinates
  int box_xmin = bottom_rois[5*n+1]; 
  int box_ymin = bottom_rois[5*n+2];
  int box_xmax = bottom_rois[5*n+3];
  int box_ymax = bottom_rois[5*n+4];
  int box_w = box_xmax-box_xmin+1.0;
  int box_h = box_ymax-box_ymin+1.0;    
  Dtype cell_w = static_cast<Dtype>(mask_width_)/static_cast<Dtype>(box_w); 
  Dtype cell_h = static_cast<Dtype>(mask_height_)/static_cast<Dtype>(box_h);

  
  //get the box in the mask
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
  const Dtype* mask_data_this = mask_data+n*mask_width_*mask_height_;
  if(y>=box_ymin && y<=box_ymax && x>=box_xmin && x<=box_xmax){
    for(int Y=Ymin; Y<=Ymax; Y++){
      for(int X=Xmin; X<=Xmax; X++){
        Dtype wty = Dtype(1.0) - fabs(static_cast<Dtype>(Y)-Ycenter)/cell_h;
        Dtype wtx = Dtype(1.0) - fabs(static_cast<Dtype>(X)-Xcenter)/cell_w;
        //wty = static_cast<Dtype>(wty>=0.5);
        //wtx = static_cast<Dtype>(wtx>=0.5);
        val += mask_data_this[Y*mask_width_ + X]*wty*wtx;
        count +=wty*wtx;

      }
    }
    cfm_data[index] = static_cast<Dtype>(val/count>=0.5);
  } 
  


  }
}
template <typename Dtype>
__global__ void CFMPoolForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const Dtype* bottom_rois,
    const Dtype* cfm_data, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_level = bottom_rois[0];
    int roi_start_w = bottom_rois[1];
    int roi_start_h = bottom_rois[2];
    int roi_end_w = bottom_rois[3];
    int roi_end_h = bottom_rois[4];

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_level * channels + c) * height * width;
    cfm_data += n*height*width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        Dtype val = bottom_data[bottom_index]*cfm_data[bottom_index];
        if (val > maxval) {
          maxval = val;
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void CFMPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  Dtype* cfm_data = cfm_weights_.mutable_gpu_data();
  const Dtype* mask_data = bottom[2]->gpu_data();
  caffe_gpu_set(cfm_weights_.count(), Dtype(0.0), cfm_data); 
  GetCFMWeights<Dtype><<<CAFFE_GET_BLOCKS(cfm_weights_.count()), CAFFE_CUDA_NUM_THREADS>>>(cfm_weights_.count(), mask_data,
  bottom_rois, cfm_data, mask_width_, mask_height_, width_, height_);
  CUDA_POST_KERNEL_CHECK;

  // NOLINT_NEXT_LINE(whitespace/operators)
  CFMPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_, height_, width_, pooled_height_,
      pooled_width_, bottom_rois, cfm_data, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
  if(top.size()>1) {
    caffe_copy(cfm_weights_.count(), cfm_weights_.gpu_data(), top[1]->mutable_gpu_data());
  }

}

template <typename Dtype>
__global__ void CFMPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, Dtype* bottom_diff, const Dtype* bottom_rois, const Dtype* cfm_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_level = offset_bottom_rois[0];
      // Skip if ROI's level doesn't match n
      if (n != roi_level) {
        continue;
      }

      int roi_start_w = offset_bottom_rois[1];
      int roi_start_h = offset_bottom_rois[2];
      int roi_end_w = offset_bottom_rois[3];
      int roi_end_h = offset_bottom_rois[4];

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(pooled_width);

      int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<Dtype>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<Dtype>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);
      Dtype wt = cfm_data[roi_n*width*height+h*width+w];
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (offset_argmax_data[ph * pooled_width + pw] == (h * width + w)) {
            gradient += offset_top_diff[ph * pooled_width + pw]*wt;
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void CFMPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  const Dtype* cfm_data = cfm_weights_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
 CFMPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), channels_,
      height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois, cfm_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(CFMPoolingLayer);

}  // namespace caffe
