#include <vector>

#include "caffe/layer.hpp"
#include "caffe/sds_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK(this->layer_param_.scale_param().has_scale())<<"Scale must be specified!";
  scale_ = (Dtype)(this->layer_param_.scale_param().scale());
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //first copy
  caffe_copy(count, bottom_data, top_data);
  //then scale
  caffe_scal(count, scale_, top_data);
  
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //first copy
    caffe_copy(count, top_diff, bottom_diff);
    //then scale
    caffe_scal(count, scale_, bottom_diff);
   
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleLayer);
#endif

INSTANTIATE_CLASS(ScaleLayer);
REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
