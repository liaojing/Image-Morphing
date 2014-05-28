#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <util/linalg.h>
#include <util/dmath.h>
#include <util/cuda_traits.h>
#include "parameters.h"
#include "pyramid.h"
#include "morph.h"

typedef rod::Matrix<float,5,5> fmat5;

void Morph::cpu_optimize_level(PyramidLevel &lvl) const
{
    int w = lvl.width;
    int h = lvl.height;
    int num = w*h;

    cv::Mat A = cv::Mat::zeros(num, num, CV_32FC1),
            Bx = cv::Mat::zeros(num, 1, CV_32FC1),
            By = cv::Mat::zeros(num, 1, CV_32FC1),
            X = cv::Mat::zeros(num, 1, CV_32FC1),
            Y = cv::Mat::zeros(num, 1, CV_32FC1);

    // setup matrix for TPS
    for(int y=0; y<h; ++y)/*{{{*/
    {
        for(int x=0; x<w; ++x)
        {
            int i = y*w + x;

            float *row = A.ptr<float>(i);

            //dxx
            if(x > 1)
            {
                row[i-2] +=  1.0f;
                row[i-1] += -2.0f;
                row[i]   +=  1.0f;
            }
            if(x > 0 && x < w-1)
            {
                row[i-1] += -2.0f;
                row[i]   +=  4.0f;
                row[i+1] += -2.0f;
            }
            if(x < w-2)
            {
                row[i]   +=  1.0f;
                row[i+1] += -2.0f;
                row[i+2] +=  1.0f;
            }

            //dy
            if(y > 1)
            {
                row[i-2*w] +=  1.0f;
                row[i-w]   += -2.0f;
                row[i]     +=  1.0f;
            }
            if(y > 0 && y< h-1)
            {
                row[i-w]  += -2.0f;
                row[i]    +=  4.0f;
                row[i+w]  += -2.0f;
            }
            if(y < h-2)
            {
                row[i]     +=  1.0f;
                row[i+w]   += -2.0f;
                row[i+2*w] +=  1.0f;
            }

            //dxy
            if(x > 0 && y > 0)
            {
                row[i-w-1] +=  2.0f;
                row[i-w]   += -2.0f;
                row[i-1]   += -2.0f;
                row[i]     +=  2.0f;
            }
            if(x < w-1 && y > 0)
            {
                row[i-w]   += -2.0f;
                row[i-w+1] +=  2.0f;
                row[i]     +=  2.0f;
                row[i+1]   += -2.0f;
            }
            if(x > 0 && y < h-1)
            {
                row[i-1]   += -2.0f;
                row[i]     +=  2.0f;
                row[i+w-1] +=  2.0f;
                row[i+w]   += -2.0f;
            }
            if(x < w-1 && y < h-1)
            {
                row[i]     +=  2.0f;
                row[i+1]   += -2.0f;
                row[i+w]   += -2.0f;
                row[i+w+1] +=  2.0f;
            }
        }
    }/*}}}*/

    // setup matrix for UI
    for(size_t i=0; i<m_params.ui_points.size(); ++i)/*{{{*/
    {
        const ConstraintPoint &cpt = m_params.ui_points[i];

        float2 p0 = make_float2(cpt.lp*make_float2((float)lvl.width, (float)lvl.height)-0.5f),
               p1 = make_float2(cpt.rp*make_float2((float)lvl.width, (float)lvl.height)-0.5f);

        float2 con = (p0+p1)/2,
               pv = (p1-p0)/2;

        for(int y=(int)floor(con.y); y<=(int)ceil(con.y); ++y)
        {
            for(int x=(int)floor(con.x); x<=(int)ceil(con.x); ++x)
            {
                if(x >=0 && x < lvl.width && y >= 0 && y < lvl.height)
                {
                    using std::abs;
                    float bilinear_w = (1 - abs(y-con.y))*(1 - abs(x-con.x));

                    int idx = y*w+x;

                    A.at<float>(idx,idx) += bilinear_w;
                    Bx.at<float>(idx,0) += bilinear_w*pv.x;
                    By.at<float>(idx,0) += bilinear_w*pv.y;
                }
            }
        }
    }/*}}}*/

    // setup border condition
    switch(m_params.bcond)/*{{{*/
    {
    case BCOND_NONE:
        break;

    case BCOND_CORNER:
        {
            int x = 0, y = 0, i = y*w + x;
            A.at<float>(i,i) += 10.f;

            x = 0, y = h-1, i = y*w + x;
            A.at<float>(i,i) += 10.f;

            x = w-1, y = h-1, i = y*w + x;
            A.at<float>(i,i) += 10.f;

            x = w-1, y = 0, i = y*w + x;
            A.at<float>(i,i) += 10.f;
        }
        break;

    case BCOND_BORDER:
        for (int x=0; x < w; ++x)
        {
            int y = 0, i = y*w + x;
            A.at<float>(i,i) += 10.f;

            y = h-1, i = y*w+x;
            A.at<float>(i,i) += 10.f;
        }

        for(int y=1; y< h-1; ++y)
        {
            int x=0, i = y*w + x;
            A.at<float>(i,i) += 10.f;

            x = w-1, i = y*w + x;
            A.at<float>(i,i) += 10.f;
        }
        break;
    }/*}}}*/

    if(!solve(A, Bx, X, cv::DECOMP_LU))
        solve(A, Bx, X, cv::DECOMP_SVD);
    if(!solve(A, By, Y, cv::DECOMP_LU))
        solve(A, By, Y, cv::DECOMP_SVD);

    std::vector<float2> v(lvl.v.size());

    for(size_t y=0; y<h; ++y)
    {
        for(size_t x=0; x<w; ++x)
        {
            int i = mem_index(lvl,make_int2(x,y));
            assert(i < v.size());

            v[i].x = X.at<float>(y*w+x,0);
            v[i].y = Y.at<float>(y*w+x,0);
        }
    }

    lvl.v.copy_from_host(&v[0], v.size());
}

