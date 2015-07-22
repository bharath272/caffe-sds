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
__global__ void LocallyConnectedForward(const int nthreads, const Dtype* bottom_data, const Dtype* label_data,
   int width, int height, int channels, int num, int grid_size,
   Dtype* temp_data, Dtype* temp2_data, Dtype* temp3_data, Dtype pos_loss_wt, Dtype neg_loss_wt, const Dtype* instance_wt)
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
  const Dtype* label_curr = label_data + n*outchannels*width*height+c*width*height;

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
  temp2_data[index]=val;
  
  Dtype label = label_data[index];
  Dtype loss = 0.0;
  loss = loss - static_cast<Dtype>(label>0.5)*pos_loss_wt*log(max(val, (Dtype)FLT_MIN));
  loss = loss - static_cast<Dtype>(label<0.5 && label>=0.0)*neg_loss_wt*log(max(1-val, (Dtype)FLT_MIN));
  /*if(label>0.0)
  {
    loss -= label>0.5?pos_loss_wt*log(max(val, (Dtype)FLT_MIN)):neg_loss_wt*log(max(1-val, (Dtype)FLT_MIN));
 
  }*/
  temp3_data[index] = loss*instance_wt[n];







  }
}
template <typename Dtype>
void LocallyConnectedWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int num=bottom[0]->num();
    const int channels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    const int grid_size = this->layer_param_.local_layer_param().grid_size();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    const Dtype* bottom_labels=bottom[1]->gpu_data();
    Dtype* temp_data = temp->mutable_gpu_data();
    Dtype* temp2_data = temp2->mutable_gpu_data();
    Dtype* temp3_data = temp3->mutable_gpu_data();
    Dtype* temp4_data = temp4->mutable_gpu_data();
    caffe_gpu_set(temp->count(), (Dtype)0.0, temp_data);
    caffe_gpu_set(temp2->count(), (Dtype)0.0, temp2_data);
    caffe_gpu_set(temp3->count(), (Dtype)1.0, temp3_data);
    caffe_gpu_set(temp4->count(), (Dtype)1.0, temp4_data);
  const Dtype neg_loss_wt = this->layer_param_.local_layer_param().neg_loss_wt();
  const Dtype pos_loss_wt = this->layer_param_.local_layer_param().pos_loss_wt();
    const int count=bottom[1]->count();
    const Dtype* instance_wt = bottom[2]->gpu_data();
    LocallyConnectedForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom_labels, width, height, 
      channels, num, grid_size, temp_data, temp2_data, temp3_data, pos_loss_wt, neg_loss_wt, instance_wt);   

    CUDA_POST_KERNEL_CHECK;
    cudaDeviceSynchronize();
    Dtype *loss = top[0]->mutable_cpu_data();
   Dtype dot=0;
    caffe_gpu_dot(count, temp4_data, temp3_data, &dot);
    loss[0] = dot/bottom[1]->num(); 
}


template <typename Dtype>
__global__ void LocallyConnectedBackward(const int nthreads, Dtype* bottom_diff, const Dtype* bottom_data, int width, int height, int channels, int num, const Dtype* label_data, const Dtype* temp_data, const Dtype* temp2_data, int grid_size, int count, const Dtype* instance_wt)
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
  Dtype label = label_data[((n*outchannels+c_out)*height+y)*width+x];
  Dtype alpha = temp_data[index];
  Dtype sumalphap = temp2_data[((n*outchannels+c_out)*height+y)*width+x];
  Dtype p = 1.0/(1.0+ exp(-bottom_data[index]));
  if(label>0.5)
  {
    Dtype frac = alpha*p/(max(sumalphap, (Dtype)FLT_MIN));
    frac = max(min(frac, Dtype(1.0)),Dtype(0.0));
    bottom_diff[index] = -frac*((Dtype)1.0 - p)*instance_wt[n]/count;

  }
  else if(label>=0)
  {
    Dtype frac = alpha*((Dtype)1.0-p)/(max((Dtype)1.0-sumalphap, (Dtype)FLT_MIN));
    frac = max(min(frac, Dtype(1.0)),Dtype(0.0));
    bottom_diff[index] = frac*p*instance_wt[n]/count;
 
  }  
  }
}
template <typename Dtype>
void LocallyConnectedWithLossLayer<Dtype>::Backward_gpu(
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
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom_diff);
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* label_data = bottom[1]->gpu_data();
    const Dtype* temp_data = temp->gpu_data();
    const Dtype* temp2_data = temp2->gpu_data();
    const Dtype* instance_wt = bottom[2]->gpu_data(); 
    LocallyConnectedBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, bottom_data, width, height, channels,num, label_data, temp_data, temp2_data, grid_size, bottom[1]->num(), instance_wt);
    CUDA_POST_KERNEL_CHECK;
    
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight, bottom_diff);
}
INSTANTIATE_LAYER_GPU_FUNCS(LocallyConnectedWithLossLayer);

}  // namespace caffe
