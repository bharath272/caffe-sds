#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sds_layers.hpp"

namespace caffe {

template<typename Dtype>
void PickSliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //the first bottom
  const Dtype* bottom_data = bottom[0]->gpu_data();
  //the second bottom
  const Dtype* bottom_indices = bottom[1]->cpu_data();
  //number, width, height and channels
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  const int inchannels = bottom[0]->channels();
  const int num = bottom[0]->num();
  //set top to 0
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0.0), top_data);
  for(int i=0;i<num; i++){
    const Dtype* bottom_indices_this = bottom_indices + i;
    const Dtype* bottom_data_this = bottom_data + i*inchannels*width*height
                                    + static_cast<int>(bottom_indices_this[0])*width*height*num_channels_;
    Dtype* top_data_this = top_data + i*num_channels_*width*height;
    caffe_copy(num_channels_*width*height, bottom_data_this, top_data_this);

  }
}

template<typename Dtype>
void PickSliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(!propagate_down[0]) { return; }
  //the first bottom
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //the second bottom
  const Dtype* bottom_indices = bottom[1]->cpu_data();
  //number, width, height and channels
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  const int inchannels = bottom[0]->channels();
  const int num = bottom[0]->num();
  //set top to 0
  const Dtype* top_diff = top[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom_diff);
  for(int i=0;i<num; i++){
    const Dtype* bottom_indices_this = bottom_indices + i;
    Dtype* bottom_diff_this = bottom_diff + i*inchannels*width*height
                                    + static_cast<int>(bottom_indices_this[0])*width*height*num_channels_;
    const Dtype* top_diff_this = top_diff + i*num_channels_*width*height;
    caffe_copy(num_channels_*width*height, top_diff_this, bottom_diff_this);

  }
  

}


INSTANTIATE_LAYER_GPU_FUNCS(PickSliceLayer);

}
