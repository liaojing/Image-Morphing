#include <util/linalg.h>
#include "stencils.h"
#include <cstdio>
#include <bitset>
#include <sstream>
#include <algorithm>
#include "pyramid.h"

typedef rod::Matrix<float,3,3> fmat3;

void calc_nb_io_stencil(PyramidLevel &lvl, rod::Matrix<imat5,5,5> &mask)
{
    mask = rod::zeros<imat5,5,5>();

    for(int i=0; i<5; ++i) // Y border
    {
        for(int j=0; j<5; ++j) // X border
        {
            for(int y=0; y<5; ++y) // Y position of current pixel
            {
                switch(i)
                {
                case 0:
                    if(y < 2)
                        continue;
                    break;
                case 1:
                    if(y < 1)
                        continue;
                    break;
                case 2:
                    break;
                case 3:
                    if(y > 3)
                        continue;
                    break;
                case 4:
                    if(y > 2)
                        continue;
                    break;
                }


                for(int x=0; x<5; ++x) // X position of current pixel
                {
                    switch(j)
                    {
                    case 0:
                        if(x < 2)
                            continue;
                        break;
                    case 1:
                        if(x < 1)
                            continue;
                        break;
                    case 2:
                        break;
                    case 3:
                        if(x > 3)
                            continue;
                        break;
                    case 4:
                        if(x > 2)
                            continue;
                        break;
                    }

                    mask[i][j][y][x] = 1;
                }
            }
        }
    }
    // just to print it out to check if everything's ok
#if 0
    for(int i=0; i<5; ++i)
    {
        for(int j=0; j<5; ++j)
        {
            printf("(%d;%d) --------------------------\n",i,j);
            for(int u=0; u<5; ++u)
            {
                for(int v=0; v<5; ++v)
                    std::cout << mask[i][j][u][v];
                printf("\n");
            }
        }
    }
#endif
}

void calc_nb_improvmask_check_stencil(PyramidLevel &lvl, rod::Matrix<imat3,5,5> &mask, rod::Matrix<int,3,3> &off)
{
    mask = rod::zeros<imat3,5,5>();

    for(int i=0; i<5; ++i) // Y position of center pixel
    {
        for(int j=0; j<5; ++j) // X position of center pixel
        {
            for(int y=0; y<5; ++y) // Y position of current pixel
            {
                for(int x=0; x<5; ++x) // X position of current pixel
                {
                    // position of current pixel relative to top left block
                    int ax = j+5 + (x-2),
                        ay = i+5 + (y-2);

                    // current block position
                    int bx = ax/5,
                        by = ay/5;

                    // position of current pixel relative to current block
                    int rx = ax - bx*5,
                        ry = ay - by*5;

                    mask[i][j][by][bx] |= (1 << (rx + ry*5)) & ((1<<25)-1);
                }
            }
        }
    }

    off = rod::zeros<int,3,3>();
    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
            off[i][j] = (i-1)*lvl.impmask_rowstride + (j-1);
    }

    // just to print it out to check if everything's ok
#if 0
    for(int i=0; i<5; ++i)
    {
        for(int j=0; j<5; ++j)
        {
            printf("(%d;%d) --------------------------\n",i,j);
            for(int u=0; u<3; ++u)
            {
                for(int l=0; l<5; ++l)
                {
                    for(int v=0; v<3; ++v)
                    {
                        std::bitset<5> b(stencil[i][j][u][v]>>(l*5));
                        std::ostringstream ss;
                        ss << b;
                        std::string x = ss.str();
                        reverse(x.begin(), x.end());
                        std::cout << x << " ";
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
    }
#endif
}

void calc_tps_stencil(rod::Matrix<fmat5,5,5> &tps)
{
#if 1
    // Standard discrete derivative masks
    fmat3 dxx = rod::zeros<float,3,3>();
    dxx[1][0] = 1; dxx[1][1] = -2; dxx[1][2] = 1;

    fmat3 dxy = rod::zeros<float,3,3>();
    dxy[0][1] = -1; dxy[0][2] = 1;
    dxy[1][1] = 1;  dxy[1][2] = -1;
#else
    // Better (according to Diego) discrete derivative masks
    fmat3 dxx;
    dxx[0][0] =  1; dxx[0][1] =  -2; dxx[0][2] =  1;
    dxx[1][0] = 10; dxx[1][1] = -20; dxx[1][2] = 10;
    dxx[2][0] =  1; dxx[2][1] =  -2; dxx[2][2] =  1;
    dxx /= 12.0f;

    fmat3 dxy = rod::zeros<float,3,3>();
    dxy[0][0] = -1; dxy[0][2] = 1;
    dxy[2][0] = 1;  dxy[2][2] = -1;
    dxy /= 4.0f;
#endif

    fmat3 dyy = transp(dxx);

    tps = rod::zeros<fmat5,5,5>();

    // mask_X[0] -> bitmask of non-zero lines of matrix X
    // mask_X[1] -> bitmask of non-zero columns of matrix X
    // ex: dxx=[0,0,0;1,-2,1;0,0,0] -> mask_dxx = {2,7}
    //     dyy=[0,1,0;0,-2,0;0,1,0] -> mask_dxx = {7,2}
    //     dxy=[0,-1,1;0,1,-1;0,0,0] -> mask_dxx = {6,3}

    unsigned char mask_dxx[2] = {0,0},
                  mask_dyy[2] = {0,0},
                  mask_dxy[2] = {0,0};

    for(int i=0; i<3; ++i)
    {
        for(int j=0; j<3; ++j)
        {
            mask_dxx[0] |= dxx[2-j][i] ? (1<<j) : 0;
            mask_dxx[1] |= dxx[i][2-j] ? (1<<j) : 0;

            mask_dyy[0] |= dyy[2-j][i] ? (1<<j) : 0;
            mask_dyy[1] |= dyy[i][2-j] ? (1<<j) : 0;

            mask_dxy[0] |= dxy[2-j][i] ? (1<<j) : 0;
            mask_dxy[1] |= dxy[i][2-j] ? (1<<j) : 0;
        }
    }


    // nesting madness...
    for(int m=0; m<5; ++m) // Y boundary
    {
        for(int n=0; n<5; ++n) // X boundary
        {
            for(int i=-1; i<=1; ++i) // mask Y pos
            {
                int ii = 2+i-1;
                for(int j=-1; j<=1; ++j) // mask X pos
                {
                    int jj = 2+j-1;
                    for(int u=0; u<3; ++u) // output Y pos
                    {
                        for(int v=0; v<3; ++v) // output X pos
                        {
                            // no values outside the boundary?
                            if((mask_dxx[0]>>(m+1+i+1))==0 && 
                                    ((mask_dxx[0]<<(5-m+1-i))&7)==0 &&
                               (mask_dxx[1]>>(n+1+j+1))==0 && 
                                    ((mask_dxx[1]<<(5-n+1-j))&7)==0)
                            {
                                tps[m][n][ii+u][jj+v] 
                                    += (dxx*(dxx[1-i][1-j]*2))[u][v];
                            }

                            // no values outside the boundary?
                            if((mask_dxy[0]>>(m+1+i+1))==0 && 
                                    ((mask_dxy[0]<<(5-m+1-i))&7)==0 &&
                               (mask_dxy[1]>>(n+1+j+1))==0 && 
                                    ((mask_dxy[1]<<(5-n+1-j))&7)==0)
                            {
                                tps[m][n][ii+u][jj+v] 
                                    += (dxy*(dxy[1-i][1-j]*4))[u][v];
                            }

                            // no values outside the boundary?
                            if((mask_dyy[0]>>(m+1+i+1))==0 && 
                                    ((mask_dyy[0]<<(5-m+1-i))&7)==0 &&
                               (mask_dyy[1]>>(n+1+j+1))==0 && 
                                    ((mask_dyy[1]<<(5-n+1-j))&7)==0)
                            {
                                tps[m][n][ii+u][jj+v] 
                                    += (dyy*(dyy[1-i][1-j]*2))[u][v];
                            }

                        }
                    }
                }
            }
        }
    }
}

