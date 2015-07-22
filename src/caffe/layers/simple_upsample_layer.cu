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
__global__ void UpsampleForward(const int nthreads, const Dtype* bottom_data,
   Dtype* top_data, int width, int height, int top_width, int top_height, int channels, int num)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int bottomtotal = width*height;
    int total = top_width*top_height;
    float H = static_cast<float>(height);
    float W = static_cast<float>(width);
    float w = static_cast<float>(top_width);
    float h = static_cast<float>(top_height);
    int x = index % top_width;
    int y = (index / top_width) % top_height;
    int c = (index / top_width / top_height) % channels;
    int n = index / top_width / top_height / channels;

    const Dtype* bottom_curr=bottom_data+n*channels*bottomtotal+c*bottomtotal;
    float Y = (static_cast<float>(y)+0.5)*H/h - 0.5;
    float Yl = floor(Y);
    float Yh = Yl+1.0;
    float wYl = Yh-Y;
    float wYh = Y-Yl;
    Yl = max(0.0f, Yl);
    Yh = min(H-1.0f, Yh);
    float X = (static_cast<float>(x)+0.5)*W/w - 0.5;
    float Xl = floor(X);
    float Xh = Xl+1.0;
    float wXl = Xh-X;
    float wXh = X-Xl;
    Xl = max(0.0f, Xl);
    Xh = min(W-1.0f, Xh);
 
    Dtype val=0.0;
    val += static_cast<Dtype>(wXl*wYl)*bottom_curr[static_cast<int>(Yl)*width + static_cast<int>(Xl)];
    val += static_cast<Dtype>(wXl*wYh)*bottom_curr[static_cast<int>(Yh)*width + static_cast<int>(Xl)];
    val += static_cast<Dtype>(wXh*wYl)*bottom_curr[static_cast<int>(Yl)*width + static_cast<int>(Xh)];
    val += static_cast<Dtype>(wXh*wYh)*bottom_curr[static_cast<int>(Yh)*width + static_cast<int>(Xh)];
    top_data[index]=val;




  }
}
template <typename Dtype>
void SimpleUpsampleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int num=bottom[0]->num();
    const int channels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    Dtype* top_data=top[0]->mutable_gpu_data();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    const int count=top[0]->count();
    UpsampleForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, width, height, top_width, top_height, 
      channels, num);     
}


template <typename Dtype>
__global__ void UpsampleBackward(const int nthreads, const Dtype* top_diff,
   Dtype* bottom_diff, int width, int height, int top_width, int top_height, int channels, int num)
{
  CUDA_KERNEL_LOOP(index, nthreads){
    int bottomtotal = width*height;
    int total = top_width*top_height;
    float H = static_cast<float>(height);
    float W = static_cast<float>(width);
    float w = static_cast<float>(top_width);
    float h = static_cast<float>(top_height);
    int X = index % width;
    int Y = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    float cell_w = w/W;
    float cell_h = h/H;
    float ycenter = (static_cast<float>(Y)+0.5)*h/H -0.5;
    int ymin = static_cast<int>(max(floor(ycenter-cell_h),0.0));
    int ymax = static_cast<int>(min(ceil(ycenter+cell_h),static_cast<double>(top_height)));
    float xcenter = (static_cast<float>(X)+0.5)*w/W -0.5;
    int xmin = static_cast<int>(max(floor(xcenter-cell_w),0.0));
    int xmax = static_cast<int>(min(ceil(xcenter+cell_w), static_cast<double>(top_width)));
    Dtype val=0;

    const Dtype* top_curr=top_diff+n*channels*total+c*total; 
    for(int y=ymin; y<ymax; y++)
    {
      float wy = 1.0 - fabs(static_cast<float>(y)-ycenter)/cell_h;
      for(int x=xmin; x<xmax; x++)
      {
         float wx = 1.0 - fabs(static_cast<float>(x)-xcenter)/cell_w;
         val += static_cast<Dtype>(wx*wy)*top_curr[y*top_width+x]; 
      }
    }
    bottom_diff[index]=val;
  }
}
template <typename Dtype>
void SimpleUpsampleLayer<Dtype>::Backward_gpu(
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
    int total=(this->top_width)*(this->top_height);
    const int bottomtotal=width*height;
    const Dtype* top_diff=top[0]->gpu_diff();
    Dtype* bottom_diff=bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    UpsampleBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_diff, width, height, top_width, top_height, 
      channels, num);     
   
}
INSTANTIATE_LAYER_GPU_FUNCS(SimpleUpsampleLayer);

}  // namespace caffe
