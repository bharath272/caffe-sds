#ifndef CAFFE_SDS_LAYERS_HPP_
#define CAFFE_SDS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {



/*
 * @brief A generic upsampling layer that upsamples to a fixed size
 */
template <typename Dtype>
class UpsampleLayer : public Layer<Dtype>{ 
  public:
  explicit UpsampleLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
   
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~UpsampleLayer(){}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const {return 1; }
  virtual inline const char* type() const { return "Upsample"; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Blob<Dtype> > temp;
  int top_height;
  int top_width;
};

/*
 * @brief A(nother) generic upsampling layer that upsamples to a fixed size, possibly also taking in a box
 */
template <typename Dtype>
class SimpleUpsampleLayer : public Layer<Dtype>{ 
  public:
  explicit SimpleUpsampleLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
   
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~SimpleUpsampleLayer(){}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const {return 1; }
  virtual inline const char* type() const { return "SimpleUpsample"; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom); 
  int top_width;
  int top_height;
};

/*
 * @brief A(nother) generic upsampling layer that upsamples to a fixed size, possibly also taking in a box
 */
template <typename Dtype>
class BoxUpsampleLayer : public Layer<Dtype>{ 
  public:
  explicit BoxUpsampleLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
   
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~BoxUpsampleLayer(){}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const {return 2; }
  virtual inline const char* type() const { return "BoxUpsample"; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom); 
  int top_width;
  int top_height;
  int numchannels;
};




/*
 */
template <typename Dtype>
class LocallyConnectedLayer : public Layer<Dtype>{ 
  public:
  explicit LocallyConnectedLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
      sigmoid_output_(new Blob<Dtype>()),
      temp(new Blob<Dtype>()),
      temp2(new Blob<Dtype>()){}
   
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~LocallyConnectedLayer(){}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const {return 1; }
  virtual inline const char* type() const { return "LocallyConnected"; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  // sigmoid_output stores the output of the sigmoid layer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  shared_ptr<Blob<Dtype> > temp;
  shared_ptr<Blob<Dtype> > temp2;

  // Vector holders to call the underlying sigmoid layer forward and backward.
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;


};
/* Local Layer
*/
template <typename Dtype>
class LocallyConnectedWithLossLayer : public LossLayer<Dtype> {
 public:
  explicit LocallyConnectedWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
          sigmoid_layer_(new SigmoidLayer<Dtype>(param)),
          sigmoid_output_(new Blob<Dtype>()),
          temp(new Blob<Dtype>()),
          temp2(new Blob<Dtype>()),
          temp3(new Blob<Dtype>()),
          temp4(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
virtual inline int ExactNumBottomBlobs() const { return 3; }

 virtual inline const char* type() const { return "LocallyConnectedWithLoss";}
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_;
  // sigmoid_output stores the output of the sigmoid layer.
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  shared_ptr<Blob<Dtype> > temp;
  shared_ptr<Blob<Dtype> > temp2;
  shared_ptr<Blob<Dtype> > temp3;
  shared_ptr<Blob<Dtype> > temp4;
 

  // Vector holders to call the underlying sigmoid layer forward and backward.
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};



template <typename Dtype>
class SuperpixelProjectionLayer : public Layer<Dtype>{ 
  public:
  explicit SuperpixelProjectionLayer(const LayerParameter& param)
    : Layer<Dtype>(param), denom(new Blob<Dtype>), temp(new Blob<Dtype>),isinitialized(false) {}  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~SuperpixelProjectionLayer(){}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const {return 3; }
  virtual inline const char* type() const { return "SuperpixelProjection"; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int numsp;
  bool isinitialized;
  shared_ptr<Blob<Dtype> > denom;
  shared_ptr<Blob<Dtype> > temp;
};

/* CFMPoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class CFMPoolingLayer : public Layer<Dtype> {
 public:
  explicit CFMPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CFMPooling"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_;
  int width_;
  int num_levels_;
  int pooled_height_;
  int pooled_width_;
  Blob<int> max_idx_;
  int mask_height_;
  int mask_width_;
  Blob<Dtype> cfm_weights_;
};

/*
* A layer to pick a slice based on the input blobs
*/

template <typename Dtype>
class PickSliceLayer : public Layer<Dtype> {
  public:
  explicit PickSliceLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PickSlice"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int num_channels_;



};
/*
* A layer to pick a set of items based on the input blobs
*/

template <typename Dtype>
class PickItemsLayer : public Layer<Dtype> {
  public:
  explicit PickItemsLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PickItems"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int num_channels_;



};


/*
* A Repmat layer
*/
template <typename Dtype>
class RepMatLayer : public Layer <Dtype> {
  public:
  explicit RepMatLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RepMat"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int rep_w;
  int rep_h;
  int rep_c;
  int rep_n;
  int bottom_w;
  int bottom_h;
  int bottom_c;
  int bottom_n;
  int top_w;
  int top_h;
  int top_c;
  int top_n;



};
/*
 * A layer to transform boxes
 */
template <typename Dtype>
class BoxTransformLayer : public Layer <Dtype> {
  explicit BoxTransformLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BoxTransform"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int feature_size_; 

};

/*
 * A layer to use the regressed boxes to produce the final box
*/
template <typename Dtype>
class UpdateBoxLayer : public Layer <Dtype> {
  explicit UpdateBoxLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UpdateBox"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 

};


/**
 * @brief Scales the input by a particular scale
 */
template <typename Dtype>
class ScaleLayer : public NeuronLayer<Dtype> {
 public:
 
 explicit ScaleLayer(const LayerParameter& param)
     : NeuronLayer<Dtype>(param) {}
 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
     const vector<Blob<Dtype>*>& top);
 virtual inline const char* type() const { return "Scale";}

 protected:
 virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 Dtype scale_;
};

/*
 * @brief Sums all bottom blobs
 */
template <typename Dtype>
class SumLayer : public Layer<Dtype>{ 
  public:
  explicit SumLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
   
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~SumLayer(){}
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MinNumBottomBlobs() const {return 1; }
  virtual inline const char* type() const { return "Sum"; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};






}
#endif
