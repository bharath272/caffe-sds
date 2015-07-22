#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sds_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {
template <typename Dtype>
void UpsampleLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  const UpsampleParameter& upsample_param = this->layer_param_.upsample_param();
  CHECK(upsample_param.has_width() && upsample_param.has_height())<<"Both width and height need to be specified.";
}
template <typename Dtype>
void UpsampleLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    this->top_width=this->layer_param_.upsample_param().width();
    this->top_height=this->layer_param_.upsample_param().height();
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), this->top_height, this->top_width);
    this->temp->Reshape(1, 8, this->top_height, this->top_width);
    
    double cell_size_h = static_cast<double>(this->top_height)/static_cast<double>(bottom[0]->height());
    double cell_size_w = static_cast<double>(this->top_width)/static_cast<double>(bottom[0]->width());
    Dtype* wtmat=this->temp->mutable_cpu_data();
    for(int i=0; i<this->top_height; i++){
     for(int j=0; j<this->top_width; j++){
        double I=(static_cast<double>(i) - cell_size_h/2)/cell_size_h;
        double J=(static_cast<double>(j) - cell_size_w/2)/cell_size_w;
        
        int lowI=static_cast<int>(floor(I));
        int lowJ=static_cast<int>(floor(J));

        int highI=lowI+1;
        int highJ=lowJ+1;
        
        double lowwtI=I-static_cast<double>(lowI);
        double lowwtJ=J-static_cast<double>(lowJ);
        double highwtI=1-lowwtI;
        double highwtJ=1-lowwtJ;

        lowI=lowI<0?0:lowI;
        lowJ=lowJ<0?0:lowJ;
        highI = (highI>=bottom[0]->height()) ? bottom[0]->height()-1 : highI;
        highJ = (highJ>=bottom[0]->width()) ? bottom[0]->width()-1 : highJ;
        wtmat[i*(this->top_width)+j]=lowI*bottom[0]->width()+lowJ;
        wtmat[(this->top_width)*(this->top_height)+i*(this->top_width)+j]=lowI*bottom[0]->width()+highJ;
        wtmat[(this->top_width)*(this->top_height)*2+i*(this->top_width)+j]=highI*bottom[0]->width()+lowJ;
        wtmat[(this->top_width)*(this->top_height)*3+i*(this->top_width)+j]=highI*bottom[0]->width()+highJ;
       
        wtmat[(this->top_width)*(this->top_height)*4+i*(this->top_width)+j]=highwtI*highwtJ;
        wtmat[(this->top_width)*(this->top_height)*5+i*(this->top_width)+j]=highwtI*lowwtJ;
        wtmat[(this->top_width)*(this->top_height)*6+i*(this->top_width)+j]=lowwtI*highwtJ;
        wtmat[(this->top_width)*(this->top_height)*7+i*(this->top_width)+j]=lowwtI*lowwtJ;

        }
    }


}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const int num=bottom[0]->num();
    const int channels=bottom[0]->channels();
    const int height=bottom[0]->height();
    const int width=bottom[0]->width();
    const Dtype* temp_data = this->temp->cpu_data();
    int total=(this->top_width)*(this->top_height);
    const int bottomtotal=width*height;
    const Dtype* lowlow=temp_data;
    const Dtype* lowhigh=temp_data+total;
    const Dtype* highlow=temp_data+total*2;
    const Dtype* highhigh=temp_data+total*3;
    
    const Dtype* lowlowwt=temp_data+total*4;
    const Dtype* lowhighwt=temp_data+total*5;
    const Dtype* highlowwt=temp_data+total*6;
    const Dtype* highhighwt=temp_data+total*7;

    Dtype* top_data=top[0]->mutable_cpu_data();
    const Dtype* bottom_data=bottom[0]->cpu_data();
    for(int n=0; n<num;n++)
    {
        for(int c=0; c<channels; c++)
        {
            const Dtype* bottom_curr=bottom_data+n*channels*bottomtotal+c*bottomtotal;
            Dtype* top_curr=top_data+n*channels*total+c*total; 
            for(int i=0; i<total; i++)
            {
                top_curr[i]=
                    bottom_curr[static_cast<int>(lowlow[i])]*lowlowwt[i] + bottom_curr[static_cast<int>(lowhigh[i])]*lowhighwt[i]
                    + bottom_curr[static_cast<int>(highlow[i])]*highlowwt[i]+bottom_curr[static_cast<int>(highhigh[i])]*highhighwt[i];
                    
            }    
        }
    }


}
template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_cpu(
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
    const Dtype* temp_data = this->temp->cpu_data();
    int total=(this->top_width)*(this->top_height);
    const int bottomtotal=width*height;
    const Dtype* lowlow=temp_data;
    const Dtype* lowhigh=temp_data+total;
    const Dtype* highlow=temp_data+total*2;
    const Dtype* highhigh=temp_data+total*3;
    
    const Dtype* lowlowwt=temp_data+total*4;
    const Dtype* lowhighwt=temp_data+total*5;
    const Dtype* highlowwt=temp_data+total*6;
    const Dtype* highhighwt=temp_data+total*7;
    
    const Dtype* top_diff=top[0]->cpu_diff();
    Dtype* bottom_diff=bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(),Dtype(0), bottom_diff);
    for(int n=0; n<num;n++)
    {
        for(int c=0; c<channels; c++)
        {
            Dtype* bottom_curr=bottom_diff+n*channels*bottomtotal+c*bottomtotal;
            const Dtype* top_curr=top_diff+n*channels*total+c*total; 
            for(int i=0; i<total; i++)
            {
                bottom_curr[static_cast<int>(lowlow[i])] += lowlowwt[i]*top_curr[i];
                bottom_curr[static_cast<int>(lowhigh[i])] += lowhighwt[i]*top_curr[i];
                bottom_curr[static_cast<int>(highlow[i])] += highlowwt[i]*top_curr[i];
                bottom_curr[static_cast<int>(highhigh[i])] += highhighwt[i]*top_curr[i];

                    
            }    
        }
    }


}
INSTANTIATE_CLASS(UpsampleLayer);
REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe
