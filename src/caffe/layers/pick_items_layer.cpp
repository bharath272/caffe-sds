#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sds_layers.hpp"

namespace caffe {

template <typename Dtype>
void PickItemsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void PickItemsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_shape = bottom[0]->shape();
  int numtopick = bottom[1]->count();
  bottom_shape[0]=numtopick;
  top[0]->Reshape(bottom_shape);

} 

template<typename Dtype>
void PickItemsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //the first bottom
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //the second bottom
  const Dtype* bottom_indices = bottom[1]->cpu_data();
  //number, width, height and channels
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int bottom_num = bottom[0]->num();
  const int top_num = bottom[1]->count();

  //set top to 0
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.0), top_data);
  for(int i=0;i<top_num; i++){
    const Dtype* bottom_indices_this = bottom_indices + i;
    const Dtype* bottom_data_this = bottom_data
                                    + static_cast<int>(bottom_indices_this[0])*channels*width*height;
    Dtype* top_data_this = top_data + i*channels*width*height;
    caffe_copy(channels*width*height, bottom_data_this, top_data_this);

  }
}

template<typename Dtype>
void PickItemsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if(!propagate_down[0]) { return; }
  //the first bottom
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  //the second bottom
  const Dtype* bottom_indices = bottom[1]->cpu_data();
  //number, width, height and channels
  const int width = bottom[0]->width();
  const int height = bottom[0]->height();
  const int channels = bottom[0]->channels();
  const int bottom_num = bottom[0]->num();
  const int top_num = bottom[1]->count();

//set top to 0
  const Dtype* top_diff = top[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);
  for(int i=0;i<top_num; i++){
    const Dtype* bottom_indices_this = bottom_indices + i;
    Dtype* bottom_diff_this = bottom_diff
                                    + static_cast<int>(bottom_indices_this[0])*channels*width*height;

    const Dtype* top_diff_this = top_diff + i*channels*width*height;

    caffe_copy(channels*width*height, top_diff_this, bottom_diff_this);

  }
  

}


#ifdef CPU_ONLY
STUB_GPU(PickItemsLayer);
#endif

INSTANTIATE_CLASS(PickItemsLayer);
REGISTER_LAYER_CLASS(PickItems);






}
