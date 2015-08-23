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
void BoxUpsampleLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  const BoxUpsampleParameter& box_upsample_param = this->layer_param_.box_upsample_param();
  CHECK(box_upsample_param.has_width() && box_upsample_param.has_height())<<"Both width and height need to be specified.";
  CHECK(box_upsample_param.has_channels())<<"num_channels need to be specified.";
}
template <typename Dtype>
void BoxUpsampleLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    this->top_width=this->layer_param_.box_upsample_param().width();
    this->top_height=this->layer_param_.box_upsample_param().height();
    this->numchannels = this->layer_param_.box_upsample_param().channels();
    //bottom[1] is [level channel_grp_id xmin ymin xmax ymax] for each box 


    top[0]->Reshape(bottom[1]->num(), this->numchannels, this->top_height, this->top_width);



}

template <typename Dtype>
void BoxUpsampleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int numlevels=bottom[0]->num();
    const int inchannels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    const int bottomtotal = width*height;

    const Dtype* boxes = bottom[1]->cpu_data();
    const int numboxes = bottom[1]->num();
    
    const int total = top_width*top_height;
    const float H = static_cast<float>(height);
    const float W = static_cast<float>(width);
    float w = static_cast<float>(top_width);
    float h = static_cast<float>(top_height);
    Dtype* top_data=top[0]->mutable_cpu_data();
    const Dtype* bottom_data=bottom[0]->cpu_data();
    for(int n=0; n<numboxes;n++)
    {
        const Dtype* box_curr = boxes + 6*n;
        int level = static_cast<int>(box_curr[0]);
        int channel_group = static_cast<int>(box_curr[1]);
        //we adapt the convention that xmin and xmax are excluded from the box. 
        Dtype box_xmin = box_curr[2];
        Dtype box_ymin = box_curr[3];
        Dtype box_xmax = box_curr[4];
        Dtype box_ymax = box_curr[5];
        Dtype box_w = box_xmax-box_xmin;
        Dtype box_h = box_ymax-box_ymin; 
        const Dtype* bottom_forthis = bottom_data+level*inchannels*bottomtotal + channel_group*numchannels*bottomtotal;
        for(int c=0; c<numchannels; c++)
        {
            const Dtype* bottom_curr=bottom_forthis + c*bottomtotal;
            Dtype* top_curr=top_data+n*numchannels*total+c*total; 
            for(int y=0; y<top_height; y++)
            {
                float Y =((static_cast<float>(y)+0.5)*box_h/h + box_ymin)*H - 0.5;
                float Yl = floor(Y);
                float Yh = Yl+1.0;
                float wYl = Yh-Y;
                float wYh = Y-Yl;
                Yl = max(Yl, 0.0f);
                Yh = min(Yh, H-1.0f);
                for(int x=0; x<top_width; x++)
                {
                   float X = ((static_cast<float>(x)+0.5)*box_w/w + box_xmin)*W- 0.5;
                   float Xl = floor(X);
                   float Xh = Xl+1.0;
                   float wXl = Xh-X;
                   float wXh = X-Xl;
                   Xl = max(Xl, 0.0f);
                   Xh = min(Xh, W-1.0f);
                   Dtype val=0.0;
                   val += static_cast<Dtype>(wXl*wYl)*bottom_curr[static_cast<int>(Yl)*width + static_cast<int>(Xl)];
                   val += static_cast<Dtype>(wXl*wYh)*bottom_curr[static_cast<int>(Yh)*width + static_cast<int>(Xl)];
                   val += static_cast<Dtype>(wXh*wYl)*bottom_curr[static_cast<int>(Yl)*width + static_cast<int>(Xh)];
                   val += static_cast<Dtype>(wXh*wYh)*bottom_curr[static_cast<int>(Yh)*width + static_cast<int>(Xh)];
                   
                   *top_curr=val;
                   top_curr++;  
                }    
            }    
        }
    }


}
template <typename Dtype>
void BoxUpsampleLayer<Dtype>::Backward_cpu(
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
    int total=(this->top_width)*(this->top_height);
    const int bottomtotal=width*height;
    const Dtype* top_diff=top[0]->cpu_diff();
    Dtype* bottom_diff=bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(),Dtype(0), bottom_diff);




    const Dtype* boxes = bottom[1]->cpu_data();
    const int numboxes = bottom[1]->num();
 


    const float H = static_cast<float>(height);
    const float W = static_cast<float>(width);
    float w = static_cast<float>(top_width);
    float h = static_cast<float>(top_height);

    for(int n=0; n<numlevels;n++)
    {
        for(int c=0; c<inchannels; c++)
        {
            Dtype* bottom_curr=bottom_diff+n*inchannels*bottomtotal+c*bottomtotal;
            for(int Y=0; Y<height; Y++)
            {
              for(int X=0; X<width; X++)
              {
                 //check each box
                for(int bid=0; bid<numboxes; bid++)
                {

                  const Dtype* box_curr = boxes + 6*bid;
                  int level = static_cast<int>(box_curr[0]);
                  
                  int channel_group = static_cast<int>(box_curr[1]);
                  if(level!=n || (c/numchannels)!=channel_group){
                    //this box either does not use this level or does not involve this channel
                    continue;
                  }
                  //we adapt the convention that xmin and xmax are excluded from the box. 
                  Dtype box_xmin = box_curr[2];
                  Dtype box_ymin = box_curr[3];
                  Dtype box_xmax = box_curr[4];
                  Dtype box_ymax = box_curr[5];
                  Dtype box_w = box_xmax-box_xmin;
                  Dtype box_h = box_ymax-box_ymin;      
                  const Dtype* top_curr=top_diff+bid*numchannels*total
                                       +(c%numchannels)*total; 
             
                  float cell_w = w/(W*box_w);
                  float cell_h = h/(H*box_h);
  
                  float ycenter = ((static_cast<float>(Y)+0.5)/H - box_ymin)*h/box_h -0.5;
                  int ymin = static_cast<int>(max(floor(ycenter-cell_h),0.0));
                  int ymax = static_cast<int>(min(ceil(ycenter+cell_h),static_cast<double>(top_height-1)));
                
                  float xcenter = ((static_cast<float>(X)+0.5)/W - box_xmin)*w/box_w -0.5;
                  int xmin = static_cast<int>(max(floor(xcenter-cell_w),0.0));
                  int xmax = static_cast<int>(min(ceil(xcenter+cell_w), static_cast<double>(top_width-1)));
                  Dtype val=0;
                  if(ymax<0 || ymin>=top_height) {continue;}
                  if(xmax<0 || xmin>=top_width) {continue;}
                  for(int y=ymin; y<ymax; y++)
                  {
                    if(y<0 || y>=top_height){
                      continue;
                    }
                    float wy = 1.0 - fabs(static_cast<float>(y)-ycenter)/cell_h;

                    //special case boundary
                    wy = ((Y==0) && y<ycenter)?1.0:wy;
                    wy = ((Y==height-1) && y>ycenter)?1.0:wy;
                    for(int x=xmin; x<xmax; x++)
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
                  *bottom_curr+= val;
                }
                
                bottom_curr++;
              }
            }    
        }
    }


}
INSTANTIATE_CLASS(BoxUpsampleLayer);
REGISTER_LAYER_CLASS(BoxUpsample);

}  // namespace caffe
