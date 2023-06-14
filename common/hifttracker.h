#ifndef HIFTTRACKER_H
#define HIFTTRACKER_H

#include <cmath>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <trttemplate.h>
#include <trttrack.h>

#include <time.h>

#include <math.h>

#define ANCH_NUM 1

struct Anchor {
    float x;
    float y;
    float w;
    float h;
};

struct Parameter {
    int exemplar_sz = 127;
    int instance_sz = 287;
//    int base_size = 8;
//    int total_stride = 8;
    int anchor_stride = 16;

//    float penalty_k = 0.055;
//    float win_influence = 0.42;
//    float lr = 0.295;
    float context_amout = 0.5;
    float penalty_k = 0.08;
    float win_influence = 0.42;
    float lr = 0.30;
    float track_w2 = 1.0;
    float track_w1 = 1.0;
    int output_size = 11;
};

template <typename t>
t x2(const cv::Rect_<t>& rect) {
    return rect.x + rect.width;
}

template <typename t>
t y2(const cv::Rect_<t>& rect) {
    return rect.y + rect.height;
}

template <typename t>
void limit(cv::Rect_<t>& rect, cv::Rect_<t> limit) {
    if (rect.x + rect.width > limit.x + limit.width) {
        rect.width = limit.x + limit.width - rect.x;
    }
    if (rect.y + rect.height > limit.y + limit.height) {
        rect.height = limit.y + limit.height - rect.y;
    }
    if (rect.x < limit.x) {
        rect.width -= (limit.x - rect.x);
        rect.x = limit.x;
    }
    if (rect.y < limit.y) {
        rect.height -= (limit.y - rect.y);
        rect.y = limit.y;
    }
    if (rect.width < 0) {
        rect.width = 0;
    }
    if (rect.height < 0) {
        rect.height = 0;
    }
}

template <typename t>
void limit(cv::Rect_<t>& rect, t width, t height, t x = 0, t y = 0) {
    limit(rect, cv::Rect_<t>(x, y, width, height));
}

template <typename t>
cv::Rect getBorder(const cv::Rect_<t>& original, cv::Rect_<t>& limited) {
    cv::Rect_<t> res;
    res.x = limited.x - original.x;
    res.y = limited.y - original.y;
    res.width = x2(original) - x2(limited);
    res.height = y2(original) - y2(limited);
    assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
    return res;
}

cv::Mat subwindowtrt(const cv::Mat& in, const cv::Rect& window, int borderType = cv::BORDER_CONSTANT);

class HiFTTracker {
   public:
    HiFTTracker(TRTTemplate *temp, TRTTrack *track);

    void init(const cv::Mat& img, cv::Rect2d& box);

//    void update(const cv::Mat& img, cv::Rect2d& box);
    void update(const cv::Mat& img);

    cv::Rect2d bbox;

   private:
    std::vector<Anchor> generate_anchor(float* output_loc_);

    std::vector<float> createPenalty(const float& target_w, const float& target_h, const std::vector<Anchor>& offsets);

    void calculateHann(const cv::Size& sz, cv::Mat& output);

    cv::Mat getSamplePatch(const cv::Mat im, const cv::Point2f posf, const int& in_sz, const int& out_sz);

    cv::Mat get_subwindow(cv::Mat im, cv::Point2f pos, int out_sz, int in_sz, cv::Scalar avg_chans);
    float z_sz_;
    float x_sz_;
    std::vector<Anchor> anchors_;
    Parameter cfg_;
    cv::Mat hann_window_;
    cv::Scalar channel_average;
    cv::Point2f pos_;

    float target_sz_w_;
    float target_sz_h_;
    float base_target_sz_w_;
    float base_target_sz_h_;
    TRTTemplate* temp_;
    TRTTrack* track_ ;

    float* hostDataBuffer1, *hostDataBuffer2, *hostDataBuffer3;

};

#endif // HIFTTRACKER_H
