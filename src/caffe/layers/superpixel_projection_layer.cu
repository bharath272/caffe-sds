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
__global__ void ProjForward(const int nthreads, const Dtype* sp, const Dtype* boxes,
   const Dtype* pred_masks, const Dtype* denom_data, Dtype* top_data, int width,
    int height, int maskw, int maskh, int numsp, int numrows) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype W = static_cast<Dtype>(width);
    Dtype H = static_cast<Dtype>(height);
    int row = index % numrows;
    int n=index / numrows;
    Dtype* topdata_this = top_data+index*numsp;
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
    
    if(row>=rounded_box_ymin && row<=rounded_box_ymax){
      Dtype y = static_cast<Dtype>(row);
      for(Dtype x=rounded_box_xmin; x<=rounded_box_xmax; x++) {
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
__global__ void ReduceForward(const int nthreads, const Dtype* temp_data, Dtype* top_data, int numsp, int numrows)
{

  CUDA_KERNEL_LOOP(index, nthreads) {
    int spid = index % numsp;
    int n=(index / numsp);
    Dtype* topdata_this = top_data+n*numsp + spid;
    const Dtype* tempdata_this = temp_data+n*numrows*numsp+spid;
    for(int i=0; i<numrows; i++)
    {
      
      topdata_this[0] += tempdata_this[0];
      tempdata_this = tempdata_this + numsp;
    }

  }
}

template <typename Dtype>
void SuperpixelProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int numboxes = bottom[1]->num();
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  int maskw = bottom[2]->width();
  int maskh = bottom[2]->height();
  Dtype W = static_cast<Dtype>(width);
  Dtype H = static_cast<Dtype>(height);
  
  const Dtype* sp = bottom[0]->gpu_data();
  const Dtype* sp_cpu = bottom[0]->cpu_data();
  const Dtype* boxes = bottom[1]->gpu_data();
  const Dtype* pred_masks = bottom[2]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* temp_data = temp->mutable_gpu_data();
  Dtype* denom_data = denom->mutable_cpu_data();
  caffe_gpu_set(top[0]->count(),ZEROD, top_data);
  caffe_gpu_set(temp->count(),ZEROD, temp_data);
  caffe_set(denom->count(),ZEROD, denom_data);
   
   
  for(int i=0; i<bottom[0]->count();i++)
  {
    Dtype spid = sp_cpu[i]-1;
    denom_data[static_cast<int>(spid)]++;
  }
  const Dtype* denom_data_gpu = denom->gpu_data();
  ProjForward<Dtype><<<CAFFE_GET_BLOCKS((bottom[1]->num())*(bottom[0]->height())), CAFFE_CUDA_NUM_THREADS>>>((bottom[1]->num())*(bottom[0]->height()), sp, boxes,
   pred_masks, denom_data_gpu,temp_data,width, height, maskw, maskh, numsp, bottom[0]->height());
  ReduceForward<Dtype><<<CAFFE_GET_BLOCKS((bottom[1]->num())*numsp), CAFFE_CUDA_NUM_THREADS>>>((bottom[1]->num())*numsp, temp_data, top_data, numsp, bottom[0]->height());
}   

template <typename Dtype>
void SuperpixelProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


}

INSTANTIATE_LAYER_GPU_FUNCS(SuperpixelProjectionLayer);

}
