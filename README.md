# SSD-Centerloss
This is an unofficial trial applying Centerloss to SSD multibox_loss function 

Technical details are in the paper: **A Discriminative Feature Learning Approach for Deep Face Recognition**
https://pan.baidu.com/s/1up_PWpR85HqVe10yhFzHoQ

SSD(Single Shot MultiBox Detector) implements the multibox_loss function in the https://github.com/weiliu89/caffe/tree/ssd. We can read the loss function through the coding multibox_loss_layer.h/multibox_loss_layer.cpp

# Motivation:
When detecting objects on the image, we often employ, including SSD, softmax function to classify the object and L1 regression to localize the object. 

$$ L(x,c,l,g) = \frac{1}{N}(L_{conf}(x,c)+ \alpha L_{loc}(x,l,g))$$

In the equation above, $L_{conf}$ represents the function that classifies each object, $L_{loc}$ stands for the localization function. $N$ is the number of default boxes. This equation means that averaging the sum of all of default boxes loss. Each default box will contribute to the final loss.

However, for some objects that are similar to each other, learning the location information may be easy. The softmax function is hard to work due to the similarity of feature of foreground samples. Center loss can effectively decrease the feature difference between the same object.
<br>
<br>
![center loss](https://img-blog.csdn.net/20180719105922908?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3piemIxMDAw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


# How to use SSD-Centerloss
### (1) add center loss layer:
Notation ~~ is the root path of your caffe-ssd dir
```
cp center_loss_layer.cpp  ~~/caffe-ssd/src/caffe/layers/
cp center_loss_layer.h  ~~/caffe-ssd/include/caffe/layers/
cp multibox_center_loss_layer.cpp ~~/caffe-ssd/src/caffe/layers/
cp multibox_center_loss_layer.hpp ~~/caffe-ssd/include/caffe/layers/
```
### (2)Then adding the following code in the caffe.proto
```
message CenterLossParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional FillerParameter center_filler = 2; // The filler for the centers
  // The first axis to be lumped into a single inner product computation;
  // all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 3 [default = 1];
}

message MultiBoxCenterLossParameter{
//center_features represents the length of features that is equal to the length of object centers in each default box.
  optional uint32 center_features = 1;
}
```
Adding  in the message LayerParameter 
```
optional MultiBoxCenterLossParameter multibox_center_loss_param = 211;//this value should be the only in this message
optional CenterLossParameter center_loss_param = 149;
```
### (3) Getting the center_features of each default box
For an instance, fc7_norm layer has 4 anchors, including 	aspect ratio = sqrt(2),1,1/2,2. Each anchor has 16 center_features. Therefore the num_output is 64.
```
layer {
  name: "fc7_norm_center_mbox_conf_new"
  type: "Convolution"
  bottom: "fc7_norm"
  top: "fc7_norm_center_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "fc7_norm_center_mbox_conf_perm"
  type: "Permute"
  bottom: "fc7_norm_center_mbox_conf"
  top: "fc7_norm_center_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "fc7_norm_mbox_center_conf_flat"
  type: "Flatten"
  bottom: "fc7_norm_center_mbox_conf_perm"
  top: "fc7_norm_mbox_center_conf_flat"
  flatten_param {
    axis: 1
  }
}
```
### (4) Changing the function type in train.prototxt
```
layer {
  name: "mbox_loss"
  type: "MultiBoxCenterLoss" # the type was changed
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  bottom: "mbox_center_conf" #mbox_center_conf is the concatenation of all the center_features in all default box.
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  propagate_down: true #center_features layers need backward.
  loss_param {
    normalization: VALID
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1
    num_classes: 21
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.2
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3
    neg_overlap: 0.1
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE
  }
  multibox_center_loss_param { 
    center_features: 16 # center_features represents the length of features that is equal to the length of object centers in each default box. 
  }
}
```