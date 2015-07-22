#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sds_layers.hpp"
#define ZEROD static_cast<Dtype>(0)
#define ceilD(a) static_cast<Dtype>(ceil(a))
#define floorD(a) static_cast<Dtype>(floor(a))
using std::max;
using std::min;
namespace caffe {
template <typename Dtype>
void SuperpixelProjectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}
template <typename Dtype>
void SuperpixelProjectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<"In Reshape";
  //Run through the first input (superpixel map) to see how many superpixels there are
  Dtype sp_max = static_cast<Dtype>(0);
  int numrows = 0;
  if(isinitialized)
  {
  const Dtype* sp_data = bottom[0]->cpu_data();
  for(int i=0; (i<bottom[0]->count()) && isinitialized; i++){
    sp_max = max(sp_max, *sp_data);
    sp_data++;

  }
  numrows = bottom[0]->height();
  } 
  isinitialized = true;
  //The size of output is numsp x numboxes
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = 1;
  top_shape[2] = bottom[1]->num();
  top_shape[3] = static_cast<int>(sp_max);
  top[0]->Reshape(top_shape);
  //LOG(INFO)<<"Reshaped top";
  top_shape[2] = 1;
  denom->Reshape(top_shape);
  
  //LOG(INFO)<<"Reshaped denom";
  top_shape[2] = numrows;
  top_shape[1] = bottom[1]->num();
  temp->Reshape(top_shape);
  //LOG(INFO)<<"Reshaped temp";

  numsp = sp_max;
  


}

template <typename Dtype>
void SuperpixelProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int numboxes = bottom[1]->num();
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  int maskw = bottom[2]->width();
  int maskh = bottom[2]->height();
  Dtype W = static_cast<Dtype>(width);
  Dtype H = static_cast<Dtype>(height);
  
  const Dtype* sp = bottom[0]->cpu_data();
  const Dtype* boxes = bottom[1]->cpu_data();
  const Dtype* pred_masks = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* denom_data = denom->mutable_cpu_data();
  caffe_set(top[0]->count(),ZEROD, top_data);
  caffe_set(denom->count(),ZEROD, denom_data);
   
   
  for(int i=0; i<bottom[0]->count();i++)
  {
    Dtype spid = sp[i]-1;
    denom_data[static_cast<int>(spid)]++;
  }



  for(int n=0; n<numboxes; n++){
    Dtype* topdata_this = top_data+n*numsp;
    const Dtype* this_box = boxes+n*6;
    const Dtype* pred_masks_this = pred_masks+n*maskw*maskh;
    Dtype box_xmin = this_box[2]*W;
    Dtype box_ymin = this_box[3]*H;
    Dtype box_xmax = this_box[4]*W;
    Dtype box_ymax = this_box[5]*H;
    Dtype box_w = box_xmax-box_xmin+1;
    Dtype box_h = box_ymax-box_ymin+1;
    Dtype rounded_box_xmin = min(max(ceilD(box_xmin),ZEROD), (W-1));
    Dtype rounded_box_ymin = min(max(ceilD(box_ymin),ZEROD), (H-1));
    Dtype rounded_box_xmax = min(max(floorD(box_xmax),ZEROD), (W-1));
    Dtype rounded_box_ymax = min(max(floorD(box_ymax),ZEROD), (H-1));
    
    for(Dtype x=rounded_box_xmin; x<=rounded_box_xmax; x++) {
      for(Dtype y=rounded_box_ymin; y<=rounded_box_ymax; y++){
        Dtype X = (x-box_xmin)/box_w*static_cast<Dtype>(maskw);
        Dtype Y = (y-box_ymin)/box_h*static_cast<Dtype>(maskh);
        X = min(max(floorD(X),ZEROD),static_cast<Dtype>(maskw-1));
        Y = min(max(floorD(Y),ZEROD), static_cast<Dtype>(maskh-1));
        Dtype spid = sp[static_cast<int>(x+y*W)]-1; 
        Dtype val =pred_masks_this[static_cast<int>(X+Y*static_cast<Dtype>(maskw))];
        topdata_this[static_cast<int>(spid)]+=val/denom_data[static_cast<int>(spid)];

      }
    }




  }


}

template <typename Dtype>
void SuperpixelProjectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


} 
#ifdef CPU_ONLY
STUB_GPU(SuperpixelProjectionLayer);
#endif

INSTANTIATE_CLASS(SuperpixelProjectionLayer);
REGISTER_LAYER_CLASS(SuperpixelProjection);

}
