#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/sds_layers.hpp"

namespace caffe {

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_gpu_scale(count, scale_, bottom_data, top_data);
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_scale(count, scale_, top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);


}  // namespace caffe
