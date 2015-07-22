// Copyright 2014 BVLC and contributors.

#include <vector>
#include<cfloat>
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
void LocallyConnectedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int grid_size = this->layer_param_.local_layer_param().grid_size();
  const int channels2 = grid_size*grid_size;
  CHECK_EQ(channels % (grid_size*grid_size), 0)<<" bottom's channels must be grid_size^2";
  temp.reset(new Blob<Dtype>(1,1,1,1));
}

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top){
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int grid_size = this->layer_param_.local_layer_param().grid_size();
  const int channels2=grid_size*grid_size;
  top[0]->Reshape(num, channels/channels2, height, width);
  temp->Reshape(bottom[0]->shape());


}

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int grid_size = this->layer_param_.local_layer_param().grid_size();
  const int channels2=grid_size*grid_size;
  const int outchannels = channels / channels2;
  const float w = static_cast<float>(width);
  const float h = static_cast<float>(height);
  float g = static_cast<float>(grid_size);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* temp_data = temp->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(temp->count(), (Dtype) 0.0, temp_data);
  
  for(int n=0; n<num; n++)
  {
    for(int c=0; c<outchannels; c++)
    {
      const Dtype* bottom_curr = bottom_data + n*channels*width*height
                                  +c*channels2*width*height;
      Dtype* temp_curr = temp_data + n*channels*width*height
                                  +c*channels2*width*height;
      Dtype* top_curr = top_data + n*outchannels*width*height+c*width*height;
      for(int y=0; y<height; y++)
      {
        float Y = (static_cast<float>(y)+0.5)*g/h - 0.5;
        float Yl = floor(Y);
        float Yh = Yl+1.0;
        float wYl = Yh-Y;
        float wYh = Y-Yl;
        Yl = max(Yl, 0.0f);
        Yh = min(Yh, g-1.0f);
 
        for(int x=0; x<width; x++)
        {
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
           f= static_cast<Dtype>(wXl*wYl)*sigmoid(bottom_curr[indexll*width*height+y*width+x]);
           val+=f;
           temp_curr[indexll*width*height+y*width+x]+=static_cast<Dtype>(wXl*wYl);
           f= static_cast<Dtype>(wXh*wYl)*sigmoid(bottom_curr[indexlh*width*height+y*width+x]);
           val+=f;
           temp_curr[indexlh*width*height+y*width+x]+=static_cast<Dtype>(wXh*wYl);

           f= static_cast<Dtype>(wXl*wYh)*sigmoid(bottom_curr[indexhl*width*height+y*width+x]);
           val+=f;
           temp_curr[indexhl*width*height+y*width+x]+=static_cast<Dtype>(wXl*wYh);


           f= static_cast<Dtype>(wXh*wYh)*sigmoid(bottom_curr[indexhh*width*height+y*width+x]);
           val+=f;
           temp_curr[indexhh*width*height+y*width+x]+=static_cast<Dtype>(wXh*wYh);

           //Save the values
           top_curr[y*width+x]=val;
            
        }
      }
    }
  }

}

template <typename Dtype>
void LocallyConnectedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int grid_size = this->layer_param_.local_layer_param().grid_size();
    const int channels2=grid_size*grid_size;
    const int outchannels = channels/channels2;   
 
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* temp_data = temp->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    for(int n=0; n<num; n++)
    {
      for(int c=0; c<channels; c++)
      {
        const Dtype* bottom_curr = bottom_data + n*channels*width*height
                                  +c*width*height;
        Dtype* bottom_diff_curr = bottom_diff + n*channels*width*height
                                  +c*width*height;

        const Dtype* temp_curr = temp_data + n*channels*width*height
                                  +c*width*height;
        int c_out = c/channels2;

        const Dtype* top_diff_curr = top_diff + n*outchannels*width*height+c_out*width*height;
        
        for(int y=0;y<height;y++)
        {
          for(int x=0;x<width;x++)
          {
             Dtype alpha = *temp_curr;
             temp_curr++;
             Dtype p = sigmoid(*bottom_curr);
             bottom_curr++;
             *bottom_diff_curr = alpha*p*(1-p)* (*top_diff_curr);
             top_diff_curr++; 
             bottom_diff_curr++;
          }
        }
      }

    } 
    //LOG(INFO)<<"maxdiff: "<<maxdiff;

  }
}

INSTANTIATE_CLASS(LocallyConnectedLayer);

REGISTER_LAYER_CLASS(LocallyConnected);

}  // namespace caffe
