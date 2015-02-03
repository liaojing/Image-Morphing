#ifndef GPUMORPH_PARAMETERS_H
#define GPUMORPH_PARAMETERS_H

#include <string>
#include <vector>
#include <vector_types.h>
#include <opencv2/core/core.hpp>

enum BoundaryCondition
{
    BCOND_NONE,
    BCOND_CORNER,
    BCOND_BORDER
};

struct ConstraintPoint
{
    double2 lp, rp;
};

struct Parameters
{
    std::string fname0, fname1;

    float w_ui, w_tps, w_ssim;
    float ssim_clamp;
    float eps;

    int max_iter;
    int start_res;
    float max_iter_drop_factor;

    bool verbose;

    BoundaryCondition bcond;

    std::vector<ConstraintPoint> ui_points;

    int ActIndex;
    int layer_num;
    cv::Mat mask1, mask2;
};

#endif

