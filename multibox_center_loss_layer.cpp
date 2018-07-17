#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "caffe/layers/multibox_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void MultiBoxCenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		if (this->layer_param_.propagate_down_size() == 0) {
			this->layer_param_.add_propagate_down(true);
			this->layer_param_.add_propagate_down(true);
			this->layer_param_.add_propagate_down(false);
			this->layer_param_.add_propagate_down(false);
			this->layer_param_.add_propagate_down(true);
		}
		const MultiBoxLossParameter& multibox_loss_param =
			this->layer_param_.multibox_loss_param();
		multibox_loss_param_ = this->layer_param_.multibox_loss_param();
		
		const MultiBoxCenterLossParameter& multibox_center_loss_param =
			this->layer_param_.multibox_center_loss_param();

		num_ = bottom[0]->num();
		num_priors_ = bottom[2]->height() / 4;
		// Get other parameters.
		CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";  
		num_classes_ = multibox_loss_param.num_classes();
		CHECK_GE(num_classes_, 1) << "num_classes should not be less than 1.";
		
		CHECK(multibox_center_loss_param.has_center_features()) << "Must provide center_features.";
		center_features_ = multibox_center_loss_param.center_features();
		CHECK_GE(center_features_, 1) << "center_features should not be less than 1.";

		share_location_ = multibox_loss_param.share_location();
		loc_classes_ = share_location_ ? 1 : num_classes_;
		match_type_ = multibox_loss_param.match_type();
		overlap_threshold_ = multibox_loss_param.overlap_threshold();
		use_prior_for_matching_ = multibox_loss_param.use_prior_for_matching();
		background_label_id_ = multibox_loss_param.background_label_id();
		use_difficult_gt_ = multibox_loss_param.use_difficult_gt();
		do_neg_mining_ = multibox_loss_param.do_neg_mining();
		neg_pos_ratio_ = multibox_loss_param.neg_pos_ratio();
		neg_overlap_ = multibox_loss_param.neg_overlap();
		code_type_ = multibox_loss_param.code_type();
		encode_variance_in_target_ = multibox_loss_param.encode_variance_in_target();
		map_object_to_agnostic_ = multibox_loss_param.map_object_to_agnostic();
		if (map_object_to_agnostic_) {
			if (background_label_id_ >= 0) {
				CHECK_EQ(num_classes_, 2);
			}
			else {
				CHECK_EQ(num_classes_, 1);
			}
		}
		//是在整个train上归一还是batch size上归一，类似于BN层
		if (!this->layer_param_.loss_param().has_normalization() &&
			this->layer_param_.loss_param().has_normalize()) {
			normalization_ = this->layer_param_.loss_param().normalize() ?
			LossParameter_NormalizationMode_VALID :
												  LossParameter_NormalizationMode_BATCH_SIZE;
		}
		else {
			normalization_ = this->layer_param_.loss_param().normalization();
		}

		if (do_neg_mining_) {
			CHECK(share_location_)
				<< "Currently only support negative mining if share_location is true.";
			CHECK_GT(neg_pos_ratio_, 0);
		}

		vector<int> loss_shape(1, 1);
		// Set up localization loss layer.
		loc_weight_ = multibox_loss_param.loc_weight();
		loc_loss_type_ = multibox_loss_param.loc_loss_type();
		// fake shape.
		vector<int> loc_shape(1, 1);
		loc_shape.push_back(4);
		loc_pred_.Reshape(loc_shape);
		loc_gt_.Reshape(loc_shape);
		loc_bottom_vec_.push_back(&loc_pred_);  //loc_bottom_vec_[0]存放loc_pred_数据
		loc_bottom_vec_.push_back(&loc_gt_);
		loc_loss_.Reshape(loss_shape);
		loc_top_vec_.push_back(&loc_loss_);
		if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_L2) {
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_l2_loc");
			layer_param.set_type("EuclideanLoss");
			layer_param.add_loss_weight(loc_weight_);
			loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
		}
		else if (loc_loss_type_ == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_smooth_L1_loc");
			layer_param.set_type("SmoothL1Loss");
			layer_param.add_loss_weight(loc_weight_);
			loc_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			loc_loss_layer_->SetUp(loc_bottom_vec_, loc_top_vec_);
		}
		else {
			LOG(FATAL) << "Unknown localization loss type.";
		}
		// Set up confidence loss layer.
		center_loss_weight_ = multibox_loss_param.center_loss_weight();
		conf_loss_type_ = multibox_loss_param.conf_loss_type();
		conf_center_bottom_vec_.push_back(&conf_center_pred_);
		conf_center_bottom_vec_.push_back(&conf_gt_);
		conf_bottom_vec_.push_back(&conf_pred_);
		conf_bottom_vec_.push_back(&conf_gt_);
		conf_loss_.Reshape(loss_shape);
		conf_top_vec_.push_back(&conf_loss_);
		conf_center_loss_.Reshape(loss_shape);
		conf_center_top_vec_.push_back(&conf_center_loss_);
		if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
			CHECK_GE(background_label_id_, 0)
				<< "background_label_id should be within [0, num_classes) for Softmax.";
			CHECK_LT(background_label_id_, num_classes_)
				<< "background_label_id should be within [0, num_classes) for Softmax.";
			//holobo:setup centerloss
			LayerParameter center_layer_param;
			center_layer_param.set_name(this->layer_param_.name() + "_center_conf");
			center_layer_param.set_type("CenterLoss");
			center_layer_param.add_loss_weight(center_loss_weight_);
			center_layer_param.mutable_loss_param()->set_normalization(
				LossParameter_NormalizationMode_NONE);
			CenterLossParameter* center_param = center_layer_param.mutable_center_loss_param();
			center_param->set_num_output(num_classes_);
			FillerParameter* filler_param = center_param->mutable_center_filler();
			filler_param->set_type("xavier");
			// Fake reshape.
			vector<int> conf_center_shape(1, 1);
			conf_gt_.Reshape(conf_center_shape);
			conf_center_shape.push_back(center_features_);
			conf_center_pred_.Reshape(conf_center_shape);
			conf_center_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(center_layer_param);
			conf_center_loss_layer_->SetUp(conf_center_bottom_vec_, conf_center_top_vec_);

			//holobo:setup SoftmaxWithLoss
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_softmax_conf");
			layer_param.set_type("SoftmaxWithLoss");          //这里可以改为center loss
			layer_param.add_loss_weight(Dtype(1.));
			layer_param.mutable_loss_param()->set_normalization(
				LossParameter_NormalizationMode_NONE);
			SoftmaxParameter* softmax_param = layer_param.mutable_softmax_param();
			softmax_param->set_axis(1);
			// Fake reshape.
			vector<int> conf_shape(1, 1);
			conf_gt_.Reshape(conf_shape);
			conf_shape.push_back(num_classes_);
			conf_pred_.Reshape(conf_shape);
			conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
		}
		else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
			LayerParameter layer_param;
			layer_param.set_name(this->layer_param_.name() + "_logistic_conf");
			layer_param.set_type("SigmoidCrossEntropyLoss");
			layer_param.add_loss_weight(Dtype(1.));
			// Fake reshape.
			vector<int> conf_shape(1, 1);
			conf_shape.push_back(num_classes_);
			conf_gt_.Reshape(conf_shape);
			conf_pred_.Reshape(conf_shape);
			conf_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
			conf_loss_layer_->SetUp(conf_bottom_vec_, conf_top_vec_);
		}
		else {
			LOG(FATAL) << "Unknown confidence loss type.";
		}
	}

	template <typename Dtype>
	void MultiBoxCenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		num_ = bottom[0]->num();
		num_priors_ = bottom[2]->height() / 4;
		num_gt_ = bottom[3]->height();
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		CHECK_EQ(bottom[0]->num(), bottom[4]->num());
		CHECK_EQ(num_priors_ * loc_classes_ * 4, bottom[0]->channels())
			<< "Number of priors must match number of location predictions.";
		CHECK_EQ(num_priors_ * num_classes_, bottom[1]->channels())
			<< "Number of priors must match number of confidence predictions.";
	}

	template <typename Dtype>
	void MultiBoxCenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* loc_data = bottom[0]->cpu_data();
		const Dtype* conf_data = bottom[1]->cpu_data();
		const Dtype* prior_data = bottom[2]->cpu_data();
		const Dtype* gt_data = bottom[3]->cpu_data();
		const Dtype* conf_center_data = bottom[4]->cpu_data(); //bottom[4] represents feature sets that calculate the center.

		// Retrieve all ground truth.
		map<int, vector<NormalizedBBox> > all_gt_bboxes;
		GetGroundTruth(gt_data, num_gt_, background_label_id_, use_difficult_gt_,
			&all_gt_bboxes);

		// Retrieve all prior bboxes. It is same within a batch since we assume all
		// images in a batch are of same dimension.
		vector<NormalizedBBox> prior_bboxes;
		vector<vector<float> > prior_variances;
		GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

		// Retrieve all predictions.
		vector<LabelBBox> all_loc_preds;
		GetLocPredictions(loc_data, num_, num_priors_, loc_classes_, share_location_,
			&all_loc_preds);

		// Find matches between source bboxes and ground truth bboxes.
		vector<map<int, vector<float> > > all_match_overlaps;
		FindMatches(all_loc_preds, all_gt_bboxes, prior_bboxes, prior_variances,
			multibox_loss_param_, &all_match_overlaps, &all_match_indices_);

		num_matches_ = 0;
		int num_negs = 0;
		// Sample hard negative (and positive) examples based on mining type.
		MineHardExamples(*bottom[1], all_loc_preds, all_gt_bboxes, prior_bboxes,
			prior_variances, all_match_overlaps, multibox_loss_param_,
			&num_matches_, &num_negs, &all_match_indices_,
			&all_neg_indices_);

		if (num_matches_ >= 1) {
			// Form data to pass on to loc_loss_layer_.
			vector<int> loc_shape(2);
			loc_shape[0] = 1;
			loc_shape[1] = num_matches_ * 4;
			loc_pred_.Reshape(loc_shape);  //loc_pred_保存：已经匹配的样本 * 4 
			loc_gt_.Reshape(loc_shape);    //同上
			Dtype* loc_pred_data = loc_pred_.mutable_cpu_data(); //读取loc_pred_数据,用指针指向
			Dtype* loc_gt_data = loc_gt_.mutable_cpu_data();   //读取gt数据
			EncodeLocPrediction(all_loc_preds, all_gt_bboxes, all_match_indices_,
				prior_bboxes, prior_variances, multibox_loss_param_,
				loc_pred_data, loc_gt_data);
			loc_loss_layer_->Reshape(loc_bottom_vec_, loc_top_vec_);
			loc_loss_layer_->Forward(loc_bottom_vec_, loc_top_vec_);
		}
		else {
			loc_loss_.mutable_cpu_data()[0] = 0;
		}

		// Form data to pass on to conf_loss_layer_.
		if (do_neg_mining_) {
			num_conf_ = num_matches_ + num_negs;
		}
		else {
			num_conf_ = num_ * num_priors_;
		}
		if (num_conf_ >= 1) {
			// Reshape the confidence data.
			vector<int> conf_shape;
			 vector<int> conf_center_shape;
			if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
				conf_shape.push_back(num_conf_);
				conf_gt_.Reshape(conf_shape);
				conf_shape.push_back(num_classes_);
				conf_pred_.Reshape(conf_shape);

				//center 
				conf_center_shape.push_back(num_conf_);
				conf_gt_.Reshape(conf_center_shape);          //conf_gt_ reshape成为[num_classes_]
				conf_center_shape.push_back(center_features_);
				conf_center_pred_.Reshape(conf_center_shape);  //conf_center_pred_ reshape成为[num_conf_,center_features]
			}
			else if (conf_loss_type_ == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
				conf_shape.push_back(1);
				conf_shape.push_back(num_conf_);
				conf_shape.push_back(num_classes_);
				conf_gt_.Reshape(conf_shape);
				conf_pred_.Reshape(conf_shape);

				//center 
				conf_center_shape.push_back(1);
				conf_center_shape.push_back(num_conf_);
				conf_center_shape.push_back(center_features_); //注意这里conf_center_shape需要成为[num_conf,center_features]
				conf_gt_.Reshape(conf_center_shape);
				conf_center_pred_.Reshape(conf_center_shape);
			}
			else {
				LOG(FATAL) << "Unknown confidence loss type.";
			}
			if (!do_neg_mining_) {
				// Consider all scores.
				// Share data and diff with bottom[1].
				CHECK_EQ(conf_pred_.count(), bottom[1]->count());
				conf_pred_.ShareData(*(bottom[1]));

				//center shares the data of bottom[4]
				CHECK_EQ(conf_center_pred_.count(), bottom[4]->count());
				conf_center_pred_.ShareData(*(bottom[4]));
			}
			Dtype* conf_pred_data = conf_pred_.mutable_cpu_data();  //conf_pred_data指针指向了bottom[1]产生的数据
			Dtype* conf_gt_data = conf_gt_.mutable_cpu_data(); //conf_pred_data指针指向了bottom[3]产生的数据
			//center shares the data of bottom[4]
			Dtype* conf_center_pred_data = conf_center_pred_.mutable_cpu_data();

			caffe_set(conf_gt_.count(), Dtype(background_label_id_), conf_gt_data);
      
			int count = 0;
			for (int i = 0; i < num_; ++i) {
				if (all_gt_bboxes.find(i) != all_gt_bboxes.end()) {
					// Save matched (positive) bboxes scores and labels.
					const map<int, vector<int> >& match_indices = all_match_indices_[i]; //all_match_indices_是上面计算的gt与prior_box的匹配结果
					for (int j = 0; j < num_priors_; ++j) {
						for (map<int, vector<int> >::const_iterator it =
							match_indices.begin(); it != match_indices.end(); ++it) {
							const vector<int>& match_index = it->second;
							CHECK_EQ(match_index.size(), num_priors_);
							if (match_index[j] == -1) {
								continue;
							}
							const int gt_label = map_object_to_agnostic_ ?
								background_label_id_ + 1 :
								all_gt_bboxes[i][match_index[j]].label();
							int idx = do_neg_mining_ ? count : j;
							switch (conf_loss_type_) {
							case MultiBoxLossParameter_ConfLossType_SOFTMAX:
								conf_gt_data[idx] = gt_label;
								break;
							case MultiBoxLossParameter_ConfLossType_LOGISTIC:
								conf_gt_data[idx * num_classes_ + gt_label] = 1;
								break;
							default:
								LOG(FATAL) << "Unknown conf loss type.";
							}
							if (do_neg_mining_) {
								// Copy scores for matched bboxes.
								caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
									conf_pred_data + count * num_classes_);  //这里conf_data就是mbox_layer的bottom[1]
								
								caffe_copy<Dtype>(center_features_, conf_center_data + j * center_features_,
									conf_center_pred_data + count * center_features_);

								++count;
							}
						}
					}
					if (do_neg_mining_) {
						// Save negative bboxes scores and labels.
						for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
							int j = all_neg_indices_[i][n];
							CHECK_LT(j, num_priors_);

							caffe_copy<Dtype>(num_classes_, conf_data + j * num_classes_,
								conf_pred_data + count * num_classes_);

							caffe_copy<Dtype>(center_features_, conf_center_data + j * center_features_,
								conf_center_pred_data + count * center_features_);

							switch (conf_loss_type_) {
							case MultiBoxLossParameter_ConfLossType_SOFTMAX:
								conf_gt_data[count] = background_label_id_;
								break;
							case MultiBoxLossParameter_ConfLossType_LOGISTIC:
								conf_gt_data[count * num_classes_ + background_label_id_] = 1;
								break;
							default:
								LOG(FATAL) << "Unknown conf loss type.";
							}
							++count;
						}
					}
				}
				// Go to next image.
				if (do_neg_mining_) {
					conf_data += bottom[1]->offset(1);
				}
				else {
					conf_gt_data += num_priors_;
				}
			}

			conf_loss_layer_->Reshape(conf_bottom_vec_, conf_top_vec_);
			conf_loss_layer_->Forward(conf_bottom_vec_, conf_top_vec_);


			conf_center_loss_layer_->Reshape(conf_center_bottom_vec_, conf_center_top_vec_);
			conf_center_loss_layer_->Forward(conf_center_bottom_vec_, conf_center_top_vec_);
		}
		else {
			conf_loss_.mutable_cpu_data()[0] = 0;
		}

		top[0]->mutable_cpu_data()[0] = 0;
		if (this->layer_param_.propagate_down(0)) {
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
				normalization_, num_, num_priors_, num_matches_);
			top[0]->mutable_cpu_data()[0] +=
				loc_weight_ * loc_loss_.cpu_data()[0] / normalizer;
		}
		if (this->layer_param_.propagate_down(1)) {
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
				normalization_, num_, num_priors_, num_matches_);
			top[0]->mutable_cpu_data()[0] += conf_loss_.cpu_data()[0] / normalizer;
			//LOG(INFO) << "conf_loss:" << conf_loss_.cpu_data()[0] / normalizer;
		}
		if (this->layer_param_.propagate_down(4))
		{
			Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
				normalization_, num_, num_priors_, num_matches_);
			top[0]->mutable_cpu_data()[0] += center_loss_weight_ * conf_center_loss_.cpu_data()[0] / normalizer;
			//LOG(INFO) << "center_loss:" << center_loss_weight_ * conf_center_loss_.cpu_data()[0] / normalizer;
		}
	}

	template <typename Dtype>
	void MultiBoxCenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		if (propagate_down[2]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to prior inputs.";
		}
		if (propagate_down[3]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}

		// Back propagate on location prediction.
		if (propagate_down[0]) {
			Dtype* loc_bottom_diff = bottom[0]->mutable_cpu_diff();
			caffe_set(bottom[0]->count(), Dtype(0), loc_bottom_diff);
			if (num_matches_ >= 1) {
				vector<bool> loc_propagate_down;
				// Only back propagate on prediction, not ground truth.
				loc_propagate_down.push_back(true);
				loc_propagate_down.push_back(false);
				loc_loss_layer_->Backward(loc_top_vec_, loc_propagate_down,
					loc_bottom_vec_);
				// Scale gradient.
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer; //top[0]->cpu_diff()[0]表示数据的diff,top[0]->cpu_diff[1]表示gt的diff
				caffe_scal(loc_pred_.count(), loss_weight, loc_pred_.mutable_cpu_diff());//这一步的作用？ 将loc_pred的梯度归一化
				// Copy gradient back to bottom[0].
				const Dtype* loc_pred_diff = loc_pred_.cpu_diff();
				int count = 0;
				for (int i = 0; i < num_; ++i) {
					for (map<int, vector<int> >::iterator it =
						all_match_indices_[i].begin();
						it != all_match_indices_[i].end(); ++it) {
						const int label = share_location_ ? 0 : it->first;
						const vector<int>& match_index = it->second;
						for (int j = 0; j < match_index.size(); ++j) {
							if (match_index[j] <= -1) {
								continue;
							}
							// Copy the diff to the right place.
							int start_idx = loc_classes_ * 4 * j + label * 4; //如果采用share_location_这里就是4*j
							caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
								loc_bottom_diff + start_idx);
							++count;
						}
					}
					loc_bottom_diff += bottom[0]->offset(1);
				}
			}
		}

		// Back propagate on confidence prediction.
		if (propagate_down[1]) {
			Dtype* conf_bottom_diff = bottom[1]->mutable_cpu_diff();
			caffe_set(bottom[1]->count(), Dtype(0), conf_bottom_diff);
			if (num_conf_ >= 1) {
				vector<bool> conf_propagate_down;
				// Only back propagate on prediction, not ground truth.
				conf_propagate_down.push_back(true);
				conf_propagate_down.push_back(false);
				conf_loss_layer_->Backward(conf_top_vec_, conf_propagate_down,
					conf_bottom_vec_);
				// Scale gradient.
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;
				caffe_scal(conf_pred_.count(), loss_weight,
					conf_pred_.mutable_cpu_diff());
				// Copy gradient back to bottom[1].
				const Dtype* conf_pred_diff = conf_pred_.cpu_diff();
				if (do_neg_mining_) {
					int count = 0;
					for (int i = 0; i < num_; ++i) {
						// Copy matched (positive) bboxes scores' diff.
						const map<int, vector<int> >& match_indices = all_match_indices_[i];
						for (map<int, vector<int> >::const_iterator it =
							match_indices.begin(); it != match_indices.end(); ++it) {
							const vector<int>& match_index = it->second;
							CHECK_EQ(match_index.size(), num_priors_);
							for (int j = 0; j < num_priors_; ++j) {
								if (match_index[j] <= -1) {
									continue;
								}
								// Copy the diff to the right place.
								caffe_copy<Dtype>(num_classes_,
									conf_pred_diff + count * num_classes_,
									conf_bottom_diff + j * num_classes_);
								++count;
							}
						}
						// Copy negative bboxes scores' diff.
						for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
							int j = all_neg_indices_[i][n];
							CHECK_LT(j, num_priors_);
							caffe_copy<Dtype>(num_classes_,
								conf_pred_diff + count * num_classes_,
								conf_bottom_diff + j * num_classes_);
							++count;
						}
						conf_bottom_diff += bottom[1]->offset(1);
					}
				}
				else {
					// The diff is already computed and stored.
					bottom[1]->ShareDiff(conf_pred_);
				}
			}
		}

		//backward for center_loss
		if (propagate_down[4]) {
			Dtype* conf_center_bottom_diff = bottom[4]->mutable_cpu_diff();
			caffe_set(bottom[4]->count(), Dtype(0), conf_center_bottom_diff);
			if (num_conf_ >= 1) {
				vector<bool> conf_propagate_down;
				// Only back propagate on prediction, not ground truth.
				conf_propagate_down.push_back(true);
				conf_propagate_down.push_back(false);

				//center calculates the gradient
				conf_center_loss_layer_->Backward(conf_center_top_vec_, conf_propagate_down,
					conf_center_bottom_vec_);
				// Scale gradient.
				Dtype normalizer = LossLayer<Dtype>::GetNormalizer(
					normalization_, num_, num_priors_, num_matches_);
				Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer;

				//center
				caffe_scal(conf_center_pred_.count(), loss_weight,
					conf_center_pred_.mutable_cpu_diff());
				// Copy gradient back to bottom[4].

				const Dtype* conf_center_pred_diff = conf_center_pred_.cpu_diff();
				if (do_neg_mining_) {
					int count = 0;
					for (int i = 0; i < num_; ++i) {
						// Copy matched (positive) bboxes scores' diff.
						const map<int, vector<int> >& match_indices = all_match_indices_[i];
						for (map<int, vector<int> >::const_iterator it =
							match_indices.begin(); it != match_indices.end(); ++it) {
							const vector<int>& match_index = it->second;
							CHECK_EQ(match_index.size(), num_priors_);
							for (int j = 0; j < num_priors_; ++j) {
								if (match_index[j] <= -1) {
									continue;
								}
								// Copy the diff to the right place.
								caffe_copy<Dtype>(center_features_,
									conf_center_pred_diff + count * center_features_,
									conf_center_bottom_diff + j * center_features_);
								++count;
							}
						}
						// Copy negative bboxes scores' diff.
						for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
							int j = all_neg_indices_[i][n];
							CHECK_LT(j, num_priors_);
							caffe_copy<Dtype>(center_features_,
								conf_center_pred_diff + count * center_features_,
								conf_center_bottom_diff + j * center_features_);
							++count;
						}
						conf_center_bottom_diff += bottom[4]->offset(1);
					}
				}
				else {
					// The diff is already computed and stored.
					bottom[4]->ShareDiff(conf_center_pred_);
				}
			}
		}
		// After backward, remove match statistics.
		all_match_indices_.clear();
		all_neg_indices_.clear();
	}

	INSTANTIATE_CLASS(MultiBoxCenterLossLayer);
	REGISTER_LAYER_CLASS(MultiBoxCenterLoss);

}  // namespace caffe
