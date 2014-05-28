#include <util/image_ops.h>
#include <opencv2/core/wimage.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "imgio.h"

template <class T>
void create_dimage(rod::dimage<T> &dimg, const cv::WImage<unsigned char> &img) /*{{{*/
{
}/*}}}*/

template <class T>
void load(rod::dimage<T> &dimg, const std::string &fname)
{
    cv::Mat temp = cv::imread(fname, -1);
    IplImage temp2 = temp;

    cv::WImageView<unsigned char> img = &temp2;

    if(img.Width()==0 || img.Height()==0)
        throw std::runtime_error(fname+": file doesn't exist or image size is zero");

    dimg.resize(img.Width(), img.Height());

    switch(img.Channels())
    {
    case 1:
        {
            rod::dimage<uchar1> temp;
            temp.copy_from_host(img(0,0), img.Width(),
                                img.Height(), img.WidthStep());
            convert(&dimg, &temp, false);
        }
        break;
    case 2:
        {
            rod::dimage<uchar2> temp;
            temp.copy_from_host((uchar2 *)img(0,0), img.Width(),
                                img.Height(), img.WidthStep()/sizeof(uchar2));
            convert(&dimg, &temp, false);
        }
        break;
    case 3:
        {
            cv::Mat temp;
            cvtColor((cv::Mat)img.Ipl(), temp, cv::COLOR_RGB2RGBA);
            IplImage temp2 = temp;

            cv::WImageViewC<unsigned char,4> temp3(&temp2);

            rod::dimage<uchar4> temp4;
            temp4.copy_from_host((uchar4 *)temp3(0,0), temp3.Width(),
                                 temp3.Height(), temp3.WidthStep()/sizeof(uchar4));
            convert(&dimg, &temp4, false);
        }
        break;
    case 4:
        {
            rod::dimage<uchar4> temp;
            temp.copy_from_host((uchar4 *)img(0,0), img.Width(),
                                img.Height(), img.WidthStep()/sizeof(uchar4));
            convert(&dimg, &temp, false);
        }
        break;
    default:
        throw std::runtime_error("Invalid number of channels");
    }
}

template void load(rod::dimage<float> &dimg, const std::string &fname);
template void load(rod::dimage<float3> &dimg, const std::string &fname);
template void load(rod::dimage<unsigned char> &dimg, const std::string &fname);
template void load(rod::dimage<uchar3> &dimg, const std::string &fname);

std::string get_format(const std::string &fname)
{
    int dot = fname.rfind('.');
    if(dot == fname.npos)
        throw std::runtime_error("Can't infer format from "+fname);

    std::string ext = fname.substr(dot+1);
    if(ext == ".jpg")
        return "jpeg";
    else
        return ext;
}

template<>
void save(const std::string &fname, const rod::dimage<float4> &img)
{
    rod::dimage<uchar4> imgaux(img.width(), img.height());
    convert(&imgaux, &img, false);

    std::vector<uchar4> cpu;
    imgaux.copy_to_host(cpu);

    cv::WImageViewC<unsigned char,4> temp((unsigned char *)&cpu[0],
                                          img.width(),img.height());

    cv::imwrite(fname, (cv::Mat)temp.Ipl());
}

template<>
void save(const std::string &fname, const rod::dimage<unsigned char> &img)
{
    std::vector<unsigned char> cpu;
    img.copy_to_host(cpu);

    cv::WImageViewC<unsigned char,1> temp((unsigned char *)&cpu[0],
                                          img.width(),img.height());
    cv::imwrite(fname, (cv::Mat)temp.Ipl());
}

template<>
void save(const std::string &fname, const rod::dimage<float3> &img)
{
    rod::dimage<uchar3> imgaux(img.width(), img.height());
    convert(&imgaux, &img, false);

    std::vector<uchar4> cpu;
    imgaux.copy_to_host(cpu);

    cv::WImageViewC<unsigned char,4> temp((unsigned char *)&cpu[0],
                                   img.width(),img.height());

    cv::Mat temp2;

    cvtColor((cv::Mat)temp.Ipl(), temp2, cv::COLOR_RGBA2RGB);

    imwrite(fname, temp2);
}

template<>
void save(const std::string &fname, const rod::dimage<uchar3> &img)
{
    std::vector<uchar4> cpu;
    img.copy_to_host(cpu);

    cv::WImageViewC<unsigned char,4> temp((unsigned char *)&cpu[0],
                                          img.width(),img.height());

    cv::Mat temp2;

    cvtColor((cv::Mat)temp.Ipl(), temp2, cv::COLOR_RGBA2RGB);

    imwrite(fname, temp2);
}

template<>
void save(const std::string &fname, const rod::dimage<float> &img)
{
    rod::dimage<uchar> imgaux(img.width(), img.height());
    convert(&imgaux, &img, false);

    std::vector<unsigned char> cpu;
    imgaux.copy_to_host(cpu);


    cv::WImageViewC<unsigned char,1> temp((unsigned char *)&cpu[0],
                                          img.width(),img.height(),img.width());
    imwrite(fname, (cv::Mat)temp.Ipl());
}
