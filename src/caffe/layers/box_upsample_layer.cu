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
__global__ void BoxUpsampleForward(const int nthreads, const Dtype* bottom_data,
   const Dtype* box_data, Dtype* top_data, int width, int height, int inchannels,
 int numlevels, int top_width, int top_height, int outchannels, int numboxes)
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
    int c = (index / top_width / top_height) % outchannels;
    int n = index / top_width / top_height / outchannels;
    
    //get box
    const Dtype* box_curr = box_data + 6*n;
    int level = static_cast<int>(box_curr[0]);
    int channel_group = static_cast<int>(box_curr[1]);
    //we adapt the convention that xmin and xmax are excluded from the box. 
    Dtype box_xmin = box_curr[2];
    Dtype box_ymin = box_curr[3];
    Dtype box_xmax = box_curr[4];
    Dtype box_ymax = box_curr[5];
    Dtype box_w = box_xmax-box_xmin;
    Dtype box_h = box_ymax-box_ymin; 
    const Dtype* bottom_curr = bottom_data+level*inchannels*bottomtotal + (channel_group*outchannels+c)*bottomtotal;



    float Y =((static_cast<float>(y)+0.5)*box_h/h + box_ymin)*H - 0.5;
    float Yl = floor(Y);
    float Yh = Yl+1.0;
    float wYl = Yh-Y;
    float wYh = Y-Yl;
    Yl = max(0.0f, Yl);
    Yh = min(H-1.0f, Yh);
    float X = ((static_cast<float>(x)+0.5)*box_w/w + box_xmin)*W- 0.5;
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
void BoxUpsampleLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int numlevels=bottom[0]->num();
    const int numboxes = bottom[1]->num();
    const int inchannels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    Dtype* top_data=top[0]->mutable_gpu_data();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    const Dtype* box_data = bottom[1]->gpu_data();
    const int count=top[0]->count();
    BoxUpsampleForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, box_data, top_data, width, height, inchannels, numlevels,top_width, top_height, numchannels, numboxes);     
}


template <typename Dtype>
__global__ void BoxUpsampleBackward(const int nthreads, const Dtype* top_diff,
   Dtype* bottom_diff, const Dtype* box_data, int width, int height, int inchannels, int numlevels, int top_width, int top_height, int outchannels, int numboxes)
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
    int c = (index / width / height) % inchannels;
    int n = index / width / height / inchannels;
    Dtype val=0;
    for(int bid=0; bid<numboxes; bid++)
    {
      const Dtype* box_curr = box_data + 6*bid;
      int level = static_cast<int>(box_curr[0]);
      
      int channel_group = static_cast<int>(box_curr[1]);
      if(level!=n || (c/outchannels)!=channel_group){
        //this box either does not use this level or does not involve this channel
        continue;
      }
      Dtype box_xmin = box_curr[2];
      Dtype box_ymin = box_curr[3];
      Dtype box_xmax = box_curr[4];
      Dtype box_ymax = box_curr[5];
      Dtype box_w = box_xmax-box_xmin;
      Dtype box_h = box_ymax-box_ymin;      
      const Dtype* top_curr=top_diff+bid*outchannels*total
                           +(c%outchannels)*total; 
      
      float cell_w = w/(W*box_w);
      float cell_h = h/(H*box_h);

      float ycenter = ((static_cast<float>(Y)+0.5)/H - box_ymin)*h/box_h -0.5;
      int ymin = static_cast<int>(max(ceil(ycenter-cell_h),0.0));
      int ymax = static_cast<int>(min(floor(ycenter+cell_h),static_cast<double>(top_height-1)));
      
      float xcenter = ((static_cast<float>(X)+0.5)/W - box_xmin)*w/box_w -0.5;
      int xmin = static_cast<int>(max(ceil(xcenter-cell_w),0.0));
      int xmax = static_cast<int>(min(floor(xcenter+cell_w), static_cast<double>(top_width-1)));
       if(ymax<0 || ymin>=top_height) {continue;}
       if(xmax<0 || xmin>=top_width) {continue;}
       for(int y=ymin; y<=ymax; y++)
       {
         if(y<0 || y>=top_height){
           continue;
         }
         float wy = 1.0 - fabs(static_cast<float>(y)-ycenter)/cell_h;

         //special case boundary
         wy = ((Y==0) && y<ycenter)?1.0:wy;
         wy = ((Y==height-1) && y>ycenter)?1.0:wy;
         for(int x=xmin; x<=xmax; x++)
         {
            if(x<0 || x>=top_width){
              continue;
            }
            float wx = 1.0 - fabs(static_cast<float>(x)-xcenter)/cell_w;
            //special case boundary
            wx = ((X==0) && x<xcenter)?1.0:wx;
            wx = ((X==width-1) && x>xcenter)?1.0:wx;
            val += static_cast<Dtype>(wx*wy)*top_curr[y*top_width+x]; 
         }
       }
      
    }
    bottom_diff[index]=val;
  }
}
template <typename Dtype>
void BoxUpsampleLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    if(!propagate_down[0])
    {
      return;
    }
    const int numlevels=bottom[0]->num();
    const int inchannels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    const int numboxes=bottom[1]->num();
    int total=(this->top_width)*(this->top_height);
    const int bottomtotal=width*height;
    const Dtype* top_diff=top[0]->gpu_diff();
    const Dtype* box_data = bottom[1]->gpu_data();
    Dtype* bottom_diff=bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    BoxUpsampleBackward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_diff, box_data, width, height, inchannels, numlevels,top_width, top_height, numchannels, numboxes);     
   
}
INSTANTIATE_LAYER_GPU_FUNCS(BoxUpsampleLayer);

}  // namespace caffe
