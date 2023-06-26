#include "hifttracker.h"
#include "cmath"
#include <fstream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

HiFTTracker::HiFTTracker(TRTTemplate* temp, TRTTrack* track)
{
    temp_ = temp;
    track_ = track;

    // create hanning window
    calculateHann(cv::Size(cfg_.output_size, cfg_.output_size), hann_window_);
}

void HiFTTracker::calculateHann(const cv::Size& sz, cv::Mat& output) {
    cv::Mat temp1(cv::Size(sz.width, 1), CV_32FC1);
    cv::Mat temp2(cv::Size(sz.height, 1), CV_32FC1);

    float* p1 = temp1.ptr<float>(0);
    float* p2 = temp2.ptr<float>(0);

    for (int i = 0; i < sz.width; ++i) p1[i] = 0.5 * (1 - cos(CV_2PI * i / (sz.width - 1)));

    for (int i = 0; i < sz.height; ++i) p2[i] = 0.5 * (1 - cos(CV_2PI * i / (sz.height - 1)));

    output = temp2.t() * temp1;
}

void HiFTTracker::init(const cv::Mat& img, cv::Rect2d& box) {
    bbox = box ;

    double w = box.width;
    double h = box.height;

    cv::Point2f init_pos;

    init_pos.x = box.x + (box.width - 1.0) / 2.0;
    init_pos.y = box.y + (box.height - 1.0) / 2.0;

    target_sz_w_ = static_cast<float>(w);
    target_sz_h_ = static_cast<float>(h);

    base_target_sz_w_ = target_sz_w_;
    base_target_sz_h_ = target_sz_h_;
    pos_ = init_pos;

//    // create hanning window
//    calculateHann(cv::Size(response_sz, response_sz), hann_window_);

    // exemplar and search sizes
    float context = (target_sz_w_ + target_sz_h_) * cfg_.context_amout;
    z_sz_ = std::sqrt((target_sz_w_ + context) * (target_sz_h_ + context));
    x_sz_ = z_sz_ * cfg_.instance_sz / cfg_.exemplar_sz;

    channel_average = cv::mean(img);
    for (int i = 0 ; i < 3 ; ++i){
        channel_average[i] = std::floor(channel_average[i]);
    }
    cv::Mat exemplar_image_patch;
//    exemplar_image_patch = getSamplePatch(img, init_pos, z_sz_, cfg_.exemplar_sz);
    exemplar_image_patch = get_subwindow(img, init_pos, cfg_.exemplar_sz, std::round(z_sz_), channel_average);

//    std::cout << "exemplar_image_patch_float" << exemplar_image_patch << std::endl;
    cv::Size size = exemplar_image_patch.size();
    int rows = size.height;
    int cols = size.width;
//    std::cout << "Rows: " << rows << ", Cols: " << cols << std::endl;

//    if (exemplar_image_patch.empty()) {
//        std::cout << "Không thể đọc hình ảnh" << std::endl;
//        return ;
//    }

//    cv::FileStorage fs("/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/image_template.yml", cv::FileStorage::WRITE);
//    if (!fs.isOpened()) {
//        std::cout << "Không thể tạo tệp YAML" << std::endl;
//        return ;
//    }

//    fs << "image" << exemplar_image_patch;

//    fs.release();

    temp_->infer(exemplar_image_patch);

    hostDataBuffer1 = temp_->output1.ptr<float>(0);
    hostDataBuffer2 = temp_->output2.ptr<float>(0);
    hostDataBuffer3 = temp_->output3.ptr<float>(0);

}

std::vector<Anchor> HiFTTracker::generate_anchor(float* output_loc_){
//    std::string file_output_txt = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/output_loc.txt";
//    std::ofstream myfile;
//    myfile.open(file_output_txt);
//    for (int i = 0 ; i < 4*cfg_.output_size*cfg_.output_size ; ++i){
//        myfile << std::to_string(output_loc_[i]) + "\n";
//        std::cout << "output_loc_ : " << output_loc_[i] << std::endl;
//    }
//    myfile.close();
    int size = cfg_.output_size;
    std::vector<float> x(size * size);
    std::vector<float> y(size * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            y[i * size + j] = cfg_.anchor_stride * i + 63 - cfg_.instance_sz / 2;
            x[i * size + j] = cfg_.anchor_stride * j + 63 - cfg_.instance_sz / 2;
        }
    }

    std::vector<int> xx(size * size);
    std::vector<int> yy(size * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            yy[i * size + j] = i;
            xx[i * size + j] = j;
        }
    }

    std::vector<float> w(size * size);
    std::vector<float> h(size * size);
    std::vector<std::vector<std::vector<float>>> shap(4, std::vector<std::vector<float>>(cfg_.output_size, std::vector<float>(cfg_.output_size)));

//    for (int i = 0; i < 400; i++){
//        std::cout << "x : " << output_loc_[i] << std::endl;
//    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < cfg_.output_size; j++) {
            for (int k = 0; k < cfg_.output_size; k++) {
                int index = i*pow(cfg_.output_size,2) + j*cfg_.output_size + k;
                if (output_loc_[index] <= -1)
                    output_loc_[index] = -0.99;
                else if (output_loc_[index] >= 1)
                    output_loc_[index] = 0.99;
//                std::cout << "i*cfg_.output_size*cfg_.output_size + j*cfg_.output_size + k : " << [i*cfg_.output_size*cfg_.output_size + j*cfg_.output_size + k] << std::endl;
                float shap_ = (log(output_loc_[index] +1) - log(1-output_loc_[index])) / 2;
                shap[i][j][k] = shap_ * 143;
//                std::cout << "shap_ : " << shap_ <<std::endl;
            }
        }
    }

    for (int i = 0; i < size * size; ++i) {
        w[i] = shap[0][yy[i]][xx[i]] + shap[1][yy[i]][xx[i]];
        h[i] = shap[2][yy[i]][xx[i]] + shap[3][yy[i]][xx[i]];
        x[i] = x[i] - shap[0][yy[i]][xx[i]] + w[i] / 2;
        y[i] = y[i] - shap[2][yy[i]][xx[i]] + h[i] / 2;
    }

    std::vector<Anchor> anchor;
    anchor.resize(size*size);

    for (int i = 0; i < size * size; ++i) {
        anchor.at(i).x = x[i];
        anchor.at(i).y = y[i];
        anchor.at(i).w = w[i];
        anchor.at(i).h = h[i];
    }

//    std::string anchor_x = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/anchor_x_cpp.txt";
//    std::string anchor_y = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/anchor_y_cpp.txt";
//    std::string anchor_w = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/anchor_w_cpp.txt";
//    std::string anchor_h = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/anchor_h_cpp.txt";
//    std::ofstream file_anchor_x;
//    std::ofstream file_anchor_y;
//    std::ofstream file_anchor_w;
//    std::ofstream file_anchor_h;
//    file_anchor_x.open(anchor_x);
//    file_anchor_y.open(anchor_y);
//    file_anchor_w.open(anchor_w);
//    file_anchor_h.open(anchor_h);
//    for (int i = 0; i < size * size; ++i) {
//        file_anchor_x << std::to_string(anchor.at(i).x) + "\n";
//        file_anchor_y << std::to_string(anchor.at(i).y) + "\n";
//        file_anchor_w << std::to_string(anchor.at(i).w) + "\n";
//        file_anchor_h << std::to_string(anchor.at(i).h) + "\n";
//    }
//    file_anchor_x.close();
//    file_anchor_y.close();
//    file_anchor_w.close();
//    file_anchor_h.close();

    return anchor;
}

void HiFTTracker::update(const cv::Mat& img) {
    cv::Mat instance_patch;
    instance_patch = get_subwindow(img, pos_, cfg_.instance_sz, std::round(x_sz_), channel_average);
//    instance_patch = getSamplePatch(img, pos_, x_sz_, cfg_.instance_sz);

//    if (instance_patch.empty()) {
//        std::cout << "Không thể đọc hình ảnh" << std::endl;
//        return ;
//    }

//    cv::FileStorage fs("/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/image_track.yml", cv::FileStorage::WRITE);
//    if (!fs.isOpened()) {
//        std::cout << "Không thể tạo tệp YAML" << std::endl;
//        return ;
//    }

//    fs << "image" << instance_patch;

//    fs.release();

    auto time_start = std::chrono::system_clock::now();
    track_->infer(instance_patch, hostDataBuffer1, hostDataBuffer2, hostDataBuffer3);
    auto time_end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = time_end - time_start;
    std::cout << "Time inference : " << elapsed_seconds.count() << std::endl;

    float* output_loc_ = track_->output_loc.ptr<float>(0);
    float* output_cls1_ = track_->output_cls1.ptr<float>(0);
    float* output_cls2_ = track_->output_cls2.ptr<float>(0);

    int size_loc = track_->size_loc;
    int size_cls2 = track_->size_cls2;

    // generate anchor
    std::vector<Anchor> reg_vec;
    reg_vec.resize(pow(cfg_.output_size,2));
    reg_vec = generate_anchor(output_loc_);

    // set penalty
    std::vector<float> penalty = createPenalty(target_sz_w_, target_sz_h_, reg_vec);
    // get response score
    // Soft max output cls
    std::vector<float> response;

    // Total cls
    for (int i = 0; i != size_cls2 ; ++i){
        float score_cls1 = std::exp(output_cls1_[size_cls2 + i]) / (std::exp(output_cls1_[i]) + std::exp(output_cls1_[size_cls2 + i]));
        float score_cls2 = output_cls2_[i];
        float score_cls = (score_cls1 * cfg_.track_w1 + score_cls2 * cfg_.track_w2)/2 ;
        response.push_back(score_cls);
    }

//    std::string file_response1 = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/response1_cpp.txt";
//    std::string file_response2 = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/response2_cpp.txt";
//    std::ofstream myfile_response1;
//    myfile_response1.open(file_response1);
//    std::ofstream myfile_response2;
//    myfile_response2.open(file_response2);
//    for (size_t i = 0; i != size_cls2; ++i) {
//        myfile_response1 << std::to_string(std::exp(output_cls1_[size_cls2 + i]) / (std::exp(output_cls1_[i]) + std::exp(output_cls1_[size_cls2 + i]))) + "\n";
//        myfile_response2 << std::to_string(output_cls2_[i]) + "\n";
//    }
//    myfile_response1.close();
//    myfile_response2.close();

//    std::string file_response = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/response_cpp.txt";
//    std::ofstream myfile_response;
//    myfile_response.open(file_response);
//    for (size_t i = 0; i != size_cls2; ++i) {
//        myfile_response << std::to_string(response[i]) + "\n";
//    }
//    myfile_response.close();


    std::vector<float> response_penalty;
    for (size_t i = 0; i < penalty.size(); ++i) {
        response_penalty.emplace_back(response[i] * penalty[i]);
//        response_penalty.emplace_back(response[i]);
    }

//    std::string file_response_penalty = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/response_penalty_cpp.txt";
//    std::ofstream myfile_response_penalty;
//    myfile_response_penalty.open(file_response_penalty);
//    for (size_t i = 0; i < penalty.size(); ++i) {
//        myfile_response_penalty << std::to_string(response_penalty[i]) + "\n";
//    }
//    myfile_response_penalty.close();


    // TODO:: response transfer to vector
    std::vector<float> response_vec;
    for (int r = 0; r < hann_window_.rows; ++r) {
        float* phann = hann_window_.ptr<float>(r);
        for (int c = 0; c < hann_window_.cols; ++c) {
            float temp =
                (1 - cfg_.win_influence) *
                    response_penalty[r * hann_window_.cols + c] +
                cfg_.win_influence * phann[c];
//            float temp = response_penalty[r * hann_window_.cols + c];
            response_vec.push_back(temp);
        }
    }

//    std::string file_response_cosin = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/response_cosin_cpp.txt";
//    std::ofstream myfile_response_cosin;
//    myfile_response_cosin.open(file_response_cosin);
//    for (size_t i = 0; i < penalty.size(); ++i) {
//        myfile_response_cosin << std::to_string(response_vec[i]) + "\n";
//    }
//    myfile_response_cosin.close();

    auto max_itr = std::max_element(response_vec.begin(), response_vec.end());
    auto id = std::distance(response_vec.begin(), max_itr);
//    std::string file_id = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/id_cpp.txt";
//    std::ofstream myfile_id;
//    myfile_id.open(file_id);
//    myfile_id << std::to_string(id) + "\n";
//    myfile_id.close();

    float offset_x = reg_vec.at(id).x * z_sz_ / cfg_.exemplar_sz;
    float offset_y = reg_vec.at(id).y * z_sz_ / cfg_.exemplar_sz;
    float offset_w = reg_vec.at(id).w * z_sz_ / cfg_.exemplar_sz;
    float offset_h = reg_vec.at(id).h * z_sz_ / cfg_.exemplar_sz;

//    std::string file_offset = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/offset_cpp.txt";
//    std::ofstream myfile_offset;
//    myfile_offset.open(file_offset);
//    myfile_offset << std::to_string(offset_x) + " ," + std::to_string(offset_y) + " ," + std::to_string(offset_w) + " ," + std::to_string(offset_h);
//    myfile_offset.close();

//    std::string file_pos = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/pos_cpp.txt";
//    std::ofstream myfile_pos;
//    myfile_pos.open(file_pos);
//    myfile_pos << std::to_string(pos_.x) + " ," + std::to_string(pos_.y) + " ," + std::to_string(target_sz_w_) + " ," + std::to_string(target_sz_h_) + "\n";
//    myfile_pos.close();

    pos_.x += offset_x;
    pos_.y += offset_y;
//    pos_.x = std::max(pos_.x, 0.f);
//    pos_.y = std::max(pos_.y, 0.f);
//    pos_.x = std::min(pos_.x, img.cols - 0.f);
//    pos_.y = std::min(pos_.y, img.rows - 0.f);
    pos_.x = std::max(pos_.x, 0.f);
    pos_.y = std::max(pos_.y, 0.f);
    pos_.x = std::min(pos_.x, static_cast<float>(img.cols));
    pos_.y = std::min(pos_.y, static_cast<float>(img.rows));

    float lr = response_penalty.at(id) * cfg_.lr;

    std::string file_lr = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/lr_cpp.txt";
    std::ofstream myfile_lr;
    myfile_lr.open(file_lr);
    myfile_lr << std::to_string(lr) + "\n";
    myfile_lr.close();

    target_sz_w_ = (1 - lr) * target_sz_w_ + lr * offset_w;
    target_sz_h_ = (1 - lr) * target_sz_h_ + lr * offset_h;

//    target_sz_w_ = std::max(target_sz_w_, 5.f);
//    target_sz_h_ = std::max(target_sz_h_, 5.f);
//    target_sz_w_ = std::min(target_sz_w_, img.cols - 1.f);
//    target_sz_h_ = std::min(target_sz_h_, img.rows - 1.f);
    target_sz_w_ = std::max(target_sz_w_, 5.f);
    target_sz_h_ = std::max(target_sz_h_, 5.f);
    target_sz_w_ = std::min(target_sz_w_, static_cast<float>(img.cols));
    target_sz_h_ = std::min(target_sz_h_, static_cast<float>(img.rows));

    // # update exemplar and instance sizes
    float context = (target_sz_w_ + target_sz_h_) / 2.0;
    z_sz_ = std::sqrt((target_sz_w_ + context) * (target_sz_h_ + context));
    x_sz_ = z_sz_ * cfg_.instance_sz / cfg_.exemplar_sz;

//    std::string file_bbox_pre = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/pre_bbox_cpp.txt";
//    std::ofstream myfile_bbox_pre;
//    myfile_bbox_pre.open(file_bbox_pre);
//    myfile_bbox_pre << std::to_string(pos_.x) + " ," + std::to_string(pos_.y) + " ," + std::to_string(target_sz_w_) + " ," + std::to_string(target_sz_h_) + "\n";
//    myfile_bbox_pre.close();

    bbox.x = pos_.x - (target_sz_w_) / 2;
    bbox.y =  pos_.y - (target_sz_h_) / 2;
    bbox.width = target_sz_w_;
    bbox.height = target_sz_h_;
//    std::string file_bbox = "/home/vee/thangkv/linhnq11/Tracking/HiFT/logs/bbox_cpp.txt";
//    std::ofstream myfile_bbox;
//    myfile_bbox.open(file_bbox);
//    myfile_bbox << std::to_string(bbox.x) + "," + std::to_string(bbox.y) + "," + std::to_string(bbox.width) + "," + std::to_string(bbox.height) + "\n";
//    myfile_bbox.close();

    // box.x = 20 ;  //pos_.x + 1 - (target_sz_w_ - 1) / 2;
    // box.y = 20 ;// pos_.y + 1 - (target_sz_h_ - 1) / 2;
    // box.width = 10;// target_sz_w_;
    // box.height = 10; //target_sz_h_;

}

std::vector<float> HiFTTracker::createPenalty(const float& target_w, const float& target_h,
                                                 const std::vector<Anchor>& offsets) {
    std::vector<float> result;

    auto padded_sz = [](const float& w, const float& h) {
        float context_tmp = 0.5 * (w + h);
        return std::sqrt((w + context_tmp) * (h + context_tmp));
    };
    auto larger_ratio = [](const float& r) { return std::max(r, 1 / r); };
    for (size_t i = 0; i < offsets.size(); ++i) {
        auto src_sz = padded_sz(target_w * cfg_.exemplar_sz / z_sz_, target_h * cfg_.exemplar_sz / z_sz_);
        auto dst_sz = padded_sz(offsets[i].w, offsets[i].h);
        auto change_sz = larger_ratio(dst_sz / src_sz);

        float src_ratio = target_w / target_h;
        float dst_ratio = offsets[i].w / offsets[i].h;
        float change_ratio = larger_ratio(dst_ratio / src_ratio);
        result.emplace_back(std::exp(-(change_ratio * change_sz - 1) * cfg_.penalty_k));
    }
    return result;
}

cv::Mat subwindowtrt(const cv::Mat& in, const cv::Rect& window, int borderType) {
    cv::Rect cutWindow = window;
    limit(cutWindow, in.cols, in.rows);

    if (cutWindow.height <= 0 || cutWindow.width <= 0) assert(0);

    cv::Rect border = getBorder(window, cutWindow);
    cv::Mat res = in(cutWindow);

    if (border != cv::Rect(0, 0, 0, 0)) {
        cv::copyMakeBorder(res, res, border.y, border.height, border.x, border.width, borderType);
    }
    return res;
}

cv::Mat HiFTTracker::getSamplePatch(const cv::Mat im, const cv::Point2f posf, const int& in_sz, const int& out_sz) {
    // Pos should be integer when input, but floor in just in case.
    cv::Point2i pos(posf.x, posf.y);
    cv::Size sample_sz = {in_sz, in_sz};  // scale adaptation
    cv::Size model_sz = {out_sz, out_sz};

    // Downsample factor
    float resize_factor = std::min(sample_sz.width / out_sz, sample_sz.height / out_sz);
    int df = std::max((float)std::floor(resize_factor - 0.1), float(1));

    cv::Mat new_im;
    im.copyTo(new_im);
    if (df > 1) {
        // compute offset and new center position
        cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
        pos.x = (pos.x - os.x - 1) / df + 1;
        pos.y = (pos.y - os.y - 1) / df + 1;
        // new sample size
        sample_sz.width = sample_sz.width / df;
        sample_sz.height = sample_sz.height / df;
        // down sample image
        int r = (im.rows - os.y) / df + 1;
        int c = (im.cols - os.x) / df;
        cv::Mat new_im2(r, c, im.type());
        new_im = new_im2;
        for (size_t i = 0 + os.y, m = 0; i < (size_t)im.rows && m < (size_t)new_im.rows; i += df, ++m) {
            for (size_t j = 0 + os.x, n = 0; j < (size_t)im.cols && n < (size_t)new_im.cols; j += df, ++n) {
                if (im.channels() == 1) {
                    new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
                } else {
                    new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
                }
            }
        }
    }

    // make sure the size is not too small and round it
    sample_sz.width = std::max(std::round(sample_sz.width), 2.0);
    sample_sz.height = std::max(std::round(sample_sz.height), 2.0);

    cv::Point pos2(pos.x - std::floor((sample_sz.width + 1) / 2), pos.y - std::floor((sample_sz.height + 1) / 2));
    cv::Mat im_patch = subwindowtrt(new_im, cv::Rect(pos2, sample_sz), cv::BORDER_REPLICATE);

    cv::Mat resized_patch;
    if (im_patch.cols == 0 || im_patch.rows == 0) {
        return resized_patch;
    }
    cv::resize(im_patch, resized_patch, model_sz);

    return resized_patch;
}

cv::Mat HiFTTracker::get_subwindow(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz, cv::Scalar avg_chans){
    int sz = original_sz;
    cv::Size im_sz = im.size();
    float center = (original_sz + 1) / 2.0;
    float context_xmin = std::floor(pos.x - center + 0.5);
    float context_xmax = context_xmin + sz - 1;
    float context_ymin = std::floor(pos.y - center + 0.5);
    float context_ymax = context_ymin + sz - 1;
    int left_pad = std::max(0, static_cast<int>(-context_xmin));
    int top_pad = std::max(0, static_cast<int>(-context_ymin));
    int right_pad = std::max(0, static_cast<int>(context_xmax - im_sz.width + 1));
    int bottom_pad = std::max(0, static_cast<int>(context_ymax - im_sz.height + 1));

    context_xmin = context_xmin + left_pad;
    context_xmax = context_xmax + left_pad;
    context_ymin = context_ymin + top_pad;
    context_ymax = context_ymax + top_pad;

//    std::cout << "im_sz.height : " << im_sz.height << " " << im_sz.width << std::endl;
//    std::cout << "left_pad : " << left_pad << " " << top_pad << " " << right_pad << " " << bottom_pad << std::endl;
//    std::cout << "context_xmin : " << context_xmin << std::endl;
//    std::cout << "im_sz : " << im_sz << std::endl;

//    cv::FileStorage fs1("/home/oem/Tracking/HiFT/im_cpp.yml", cv::FileStorage::WRITE);
//    if (!fs1.isOpened()) {
//        std::cout << "Không thể tạo tệp YAML" << std::endl;
//    }

//    fs1 << "image" << im;

    int r = im.rows;
    int c = im.cols;
    int k = im.channels();
    cv::Mat im_patch;
//    std::cout << "avg_chans " << avg_chans[0] << " " << avg_chans[1] << std::endl;

    if (top_pad || bottom_pad || left_pad || right_pad) {

        cv::Size size(c + left_pad + right_pad, r + top_pad + bottom_pad);
//        std::cout << "size : "<< size << std::endl;
        cv::Mat te_im(size, CV_8UC(k), cv::Scalar(0, 0, 0));
//        std::cout << "left_pad, top_pad , c, r : " << left_pad << " " << top_pad <<" " << c <<" "<< r << std::endl;
        im.copyTo(te_im(cv::Rect(left_pad, top_pad, c, r)));
//        cv::FileStorage fs2("/home/oem/Tracking/HiFT/roi_cpp.yml", cv::FileStorage::WRITE);
//        if (!fs2.isOpened()) {
//            std::cout << "Không thể tạo tệp YAML" << std::endl;
//        }

//        fs2 << "image" << roi;

//        fs2.release();
        if (top_pad) {
            te_im(cv::Rect(left_pad, 0, c, top_pad)) = avg_chans;
        }
        if (bottom_pad) {
            te_im(cv::Rect(left_pad, r + top_pad, c, te_im.rows - (r + top_pad))) = avg_chans;
        }
        if (left_pad) {
            te_im(cv::Rect(0, 0, left_pad, te_im.rows)) = avg_chans;
        }
        if (right_pad) {
//            std::cout << "right_pad" << " : " << cv::Rect(c + left_pad, 0, right_pad, size.height) << std::endl;
            te_im(cv::Rect(c + left_pad, 0, te_im.cols - (c + left_pad), te_im.rows)) = avg_chans;
////            cv::Mat roi_right(te_im, cv::Rect(c + left_pad, 0, 0, size.height));
////            assign_avgchans(roi_right,avg_chans);
//            cv::FileStorage fs3("/home/oem/Tracking/HiFT/roiright_cpp.yml", cv::FileStorage::WRITE);
//            if (!fs3.isOpened()) {
//                std::cout << "Không thể tạo tệp YAML" << std::endl;
//            }

//            fs3 << "image" << te_im(cv::Rect(c + left_pad, 0, te_im.cols - (c + left_pad), te_im.rows));

//            fs3.release();
        }

        cv::Rect roi_rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1);
        im_patch = te_im(roi_rect);
    }
    else {
        im_patch = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }

//    std::cout << "im_patch.size " << im_patch.size() << std::endl;
    if (model_sz != original_sz) {
        cv::resize(im_patch, im_patch, cv::Size(model_sz, model_sz));
    }

//    cv::FileStorage fs("/home/oem/Tracking/HiFT/crop_cpp.yml", cv::FileStorage::WRITE);
//    if (!fs.isOpened()) {
//        std::cout << "Không thể tạo tệp YAML" << std::endl;
//    }

//    fs << "image" << im_patch;

//    fs.release();

//    getchar();

    return im_patch;
}
