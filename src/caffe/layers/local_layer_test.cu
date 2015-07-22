#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sds_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
namespace caffe {
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}
template <typename Dtype>
__global__ void LocallyConnectedForward2(const int nthreads, const Dtype* bottom_data,
   int width, int height, int channels, int num, int grid_size,
   Dtype* temp_data, Dtype* top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
  int channels2 = grid_size*grid_size;
  int outchannels = channels/channels2;
  float w = static_cast<float>(width);
  float h = static_cast<float>(height);
  float g = static_cast<float>(grid_size);
  
  int x = index % width;
  int y = (index / width) % height;
  int c = (index / width / height) % outchannels;
  int n = index / width / height / outchannels;
  
  const Dtype* bottom_curr = bottom_data + n*channels*width*height
                                  +c*channels2*width*height;
  Dtype* temp_curr = temp_data + n*channels*width*height
                              +c*channels2*width*height;

  float Y = (static_cast<float>(y)+0.5)*g/h - 0.5;
  float Yl = floor(Y);
  float Yh = Yl+1.0;
  float wYl = Yh-Y;
  float wYh = Y-Yl;
  Yl = max(Yl, 0.0f);
  Yh = min(Yh, g-1.0f);
  float X = (static_cast<float>(x)+0.5)*g/w - 0.5;
  float Xl = floor(X);
  float Xh = Xl+1.0;
  float wXl = Xh-X;
  float wXh = X-Xl;
  Xl = max(Xl, 0.0f);
  Xh = min(Xh, g-1.0f);
  //compute the index from which to pull
  int indexll = static_cast<int>(Yl*g+Xl);
  int indexlh = static_cast<int>(Yl*g+Xh);
  int indexhl = static_cast<int>(Yh*g+Xl);
  int indexhh = static_cast<int>(Yh*g+Xh);
  //let's pick all the channels
  Dtype val=0.0;
  Dtype f;
  f= static_cast<Dtype>(wXl*wYl)/(1. + exp(-bottom_curr[indexll*width*height+y*width+x]));
  val+=f;
  temp_curr[indexll*width*height+y*width+x] += static_cast<Dtype>(wXl*wYl);
  f= static_cast<Dtype>(wXh*wYl)/(1. + exp(-bottom_curr[indexlh*width*height+y*width+x]));
  val+=f;
  temp_curr[indexlh*width*height+y*width+x] +=static_cast<Dtype>(wXh*wYl);

  f= static_cast<Dtype>(wXl*wYh)/(1. + exp(-bottom_curr[indexhl*width*height+y*width+x]));
  val+=f;
  temp_curr[indexhl*width*height+y*width+x] +=static_cast<Dtype>(wXl*wYh);


  f= static_cast<Dtype>(wXh*wYh)/(1. + exp(-bottom_curr[indexhh*width*height+y*width+x]));
  val+=f;
  temp_curr[indexhh*width*height+y*width+x] +=static_cast<Dtype>(wXh*wYh);

  //Save the values
  top_data[index]=val;







  }
}
template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int num=bottom[0]->num();
    const int channels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    const int grid_size = this->layer_param_.local_layer_param().grid_size();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* temp_data = temp->mutable_gpu_data();
    caffe_gpu_set(temp->count(), (Dtype)0.0, temp_data);
    
    const int count=top[0]->count();
    LocallyConnectedForward2<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, width, height, 
      channels, num, grid_size, temp_data, top_data);    
    
    
    
}


template <typename Dtype>
__global__ void LocallyConnectedBackward2(const int nthreads, Dtype* bottom_diff, const Dtype* bottom_data, int width, int height, int channels, int num,  const Dtype* temp_data,  const Dtype* top_diff, int grid_size)
{
  CUDA_KERNEL_LOOP(index, nthreads){
  int channels2 = grid_size*grid_size;
  int outchannels = channels/channels2;
  float w = static_cast<float>(width);
  float h = static_cast<float>(height);
  float g = static_cast<float>(grid_size);
  
  int x = index % width;
  int y = (index / width) % height;
  int c = (index / width / height) % channels;
  int n = index / width / height / channels;
  int c_out = c/channels2;
  Dtype alpha = temp_data[index];
  Dtype p = 1.0/(1.0+ exp(-bottom_data[index]));
  bottom_diff[index] = alpha*p*(1-p)*top_diff[n*outchannels*width*height + c_out*width*height + y*width + x];  

  }
}
template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if(!propagate_down[0])
    {
      return;
    }
    const int num=bottom[0]->num();
    const int channels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    const int grid_size = this->layer_param_.local_layer_param().grid_size();
    Dtype* bottom_diff=bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* temp_data = temp->gpu_data();
    LocallyConnectedBackward2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, bottom_data, width, height, channels,num, temp_data, top_diff, grid_size);
  
}
INSTANTIATE_LAYER_GPU_FUNCS(LocallyConnectedLayer);

}  // namespace caffe
