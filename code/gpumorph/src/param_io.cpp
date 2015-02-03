#include <libxml/xpath.h>
#include <vector_functions.h>
#include <stdexcept>
#include <sstream>
#include <typeinfo>
#include "parameters.h"

namespace {

std::string get_data(xmlXPathContextPtr ctx, const std::string &addr,
                     const std::string &def)
{
    xmlXPathObject *xpath = NULL;
    xmlChar *str = NULL;

    try
    {
        xpath = xmlXPathEvalExpression(BAD_CAST addr.c_str(), ctx);
        if(!xpath)
            throw std::runtime_error("Unable to evaluate xpath expression");

        if(xmlXPathNodeSetIsEmpty(xpath->nodesetval))
        {
            xmlXPathFreeObject(xpath);
            return def;
        }

        str = xmlXPathCastToString(xpath);
        if(!str)
        {
            throw std::runtime_error("Error converting "+ std::string((char *)str) + " to string");
        }

        std::string ret = (char *)str;

        xmlFree(str);
        xmlXPathFreeObject(xpath);

        return ret;
    }
    catch(...)
    {
        if(str)
            xmlFree(str);

        if(xpath)
            xmlXPathFreeObject(xpath);

        throw;
    }
}

std::string get_data(xmlXPathContextPtr ctx, const std::string &addr,
                     const char *def)
{
    return get_data(ctx, addr, std::string(def));
}

template <class T>
T get_data(xmlXPathContextPtr ctx, const std::string &addr, const T &def)
{
    std::ostringstream ssdef;
    ssdef << def;

    std::istringstream ss(get_data(ctx, addr, ssdef.str()));

    T ret;
    ss >> ret;
    if(!ss)
        throw std::runtime_error("Error converting "+ss.str()+" to '"+typeid(T).name());

    return ret;
}

} // local namespace

void parse_config_xml(Parameters &params, const std::string &fname)
{
    static bool initxml = false;
    if(!initxml)
    {
        xmlInitParser();
        xmlXPathInit();
        atexit(xmlCleanupParser);

        initxml = true;
    }

    xmlDoc *doc = NULL;
    xmlXPathContext *ctx = NULL;

    try
    {
        doc = xmlParseFile(fname.c_str());
        if(!doc)
            throw std::runtime_error("Syntax error in "+fname);

        if(!doc->children)
            throw std::runtime_error("Semantic error in "+fname);

        ctx = xmlXPathNewContext(doc);
        if(!ctx)
            throw std::runtime_error("Unable to create XPath context");

        params.fname0 = get_data(ctx, "//project/images/@image1",params.fname0);
        params.fname1 = get_data(ctx, "//project/images/@image2",params.fname1);

        // circumvent MdiEditor's notion that relative files must begin with '/'
        if(!params.fname0.empty() && (params.fname0[0] == '/' || params.fname0[0] == '\\'))
            params.fname0 = params.fname0.substr(1);
        if(!params.fname1.empty() && (params.fname1[0] == '/' || params.fname1[0]=='\\'))
            params.fname1 = params.fname1.substr(1);

	int slash = fname.find_last_of("/\\");
	std::string root_path;

	if(slash != fname.npos)
	{
	    root_path = fname.substr(0,slash+1); // root_path includes final slash
	    params.fname0 = root_path + params.fname0;
	    params.fname1 = root_path + params.fname1;
	}

        std::string base = "/project/layers";

        params.w_tps
            = get_data(ctx, base+"/l0/parameters/weight/@tps", params.w_tps);

        params.w_ssim
            = get_data(ctx, base+"/l0/parameters/weight/@ssim", params.w_ssim);

        params.w_ui
            = get_data(ctx, base+"/l0/parameters/weight/@ui", params.w_ui);

        params.ssim_clamp
            = 1-get_data(ctx, base+"/l0/parameters/weight/@ssimclamp", 1-params.ssim_clamp);

        int bound;
        switch(params.bcond)
        {
        case BCOND_NONE:
            bound = 0;
            break;
        case BCOND_CORNER:
            bound = 1;
            break;
        case BCOND_BORDER:
            bound = 2;
            break;
        }

        bound = get_data(ctx, base+"/l0/parameters/boundary/@lock", bound);

        switch(bound)
        {
        case 0:
            params.bcond = BCOND_NONE;
            break;
        case 1:
            params.bcond = BCOND_CORNER;
            break;
        case 2:
            params.bcond = BCOND_BORDER;
            break;
        default:
            throw std::runtime_error("Bad boundary value");
        }

        params.eps
            = get_data(ctx, base+"/l0/parameters/debug/@eps", params.eps);
        params.start_res
            = get_data(ctx, base+"/l0/parameters/debug/@startres", params.start_res);
        params.max_iter
            = get_data(ctx, base+"/l0/parameters/debug/@iternum", params.max_iter);
        params.max_iter_drop_factor
            = get_data(ctx, base+"/l0/parameters/debug/@dropfactor", params.max_iter_drop_factor);

        std::string pts0
            = get_data(ctx, base+"/l0/parameters/points/@image1", "");

        std::string pts1
            = get_data(ctx, base+"/l0/parameters/points/@image2", "");

        if(!pts0.empty())
        {
            params.ui_points.clear();

            std::istringstream ss0(pts0), ss1(pts1);

            while(ss0 && ss1)
            {
                ConstraintPoint cpt;

                float2 pt;
                ss0 >> pt.x >> pt.y;
                cpt.lp = make_double2(pt.x, pt.y);

                ss1 >> pt.x >> pt.y;
                cpt.rp = make_double2(pt.x, pt.y);

                if(ss0 && ss1)
                    params.ui_points.push_back(cpt);
            }

            if(ss0.eof() && !ss1.eof() || !ss0.eof() && ss1.eof())
                throw std::runtime_error("Control point parsing error");
        }

        xmlXPathFreeContext(ctx);
        xmlFreeDoc(doc);
    }
    catch(...)
    {
        if(ctx)
            xmlXPathFreeContext(ctx);
        if(doc)
            xmlFreeDoc(doc);
        throw;
    }
}
