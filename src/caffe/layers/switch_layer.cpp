#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom.size(), 3);
  CHECK_EQ(top.size(), 1);
  num_ = bottom[1]->num();
  channels_ = bottom[1]->channels();
  height_ = bottom[1]->height();
  width_ = bottom[1]->width();
  for (int i = 2; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->num(), num_);
    CHECK_EQ(bottom[i]->channels(), channels_);
    CHECK_EQ(bottom[i]->height(), height_);
    CHECK_EQ(bottom[i]->width(), width_);
  }
//  CHECK_EQ(top[0]->num(), num_);
//  CHECK_EQ(top[0]->channels(), channels_);
//  CHECK_EQ(top[0]->height(), height_);
//  CHECK_EQ(top[0]->width(), width_);
  CHECK_EQ(bottom[0]->num(), num_);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* switch_data = bottom[0]->cpu_data();
  const int num_elem = channels_ * height_ * width_;
  for (int i = 0; i < num_; ++i) {
    CHECK_EQ(switch_data[i], std::ceil(switch_data[i]));
    const int bottom_data_id = static_cast<int>(switch_data[i]);
    CHECK_GE(bottom_data_id, 0);
    CHECK_LT(bottom_data_id, bottom.size() - 1);
    const Dtype* bottom_data = bottom[bottom_data_id + 1]->cpu_data();
    caffe_copy(
        num_elem,
        bottom_data + bottom[bottom_data_id + 1]->offset(i),
        top_data + top[0]->offset(i));
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int num_elem = channels_ * height_ * width_;
  const Blob<Dtype>* top_blob = top[0];
  const Dtype* top_diff = top_blob->cpu_diff();
  const Dtype* switch_data = bottom[0]->cpu_data();
  for (int i = 0; i < num_; ++i) {
    CHECK_EQ(switch_data[i], std::ceil(switch_data[i]));
    const int bottom_data_id = static_cast<int>(switch_data[i]);
    if (!propagate_down[bottom_data_id + 1]) { continue; }
    Blob<Dtype>* bottom_blob = bottom[bottom_data_id + 1];
    Dtype* bottom_diff = bottom_blob->mutable_cpu_diff();
    caffe_copy(
        num_elem,
        top_diff + top_blob->offset(i),
        bottom_diff + bottom_blob->offset(i));
  }
}

INSTANTIATE_CLASS(SwitchLayer);

}  // namespace caffe



