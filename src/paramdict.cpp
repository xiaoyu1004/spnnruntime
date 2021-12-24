#include "log.h"
#include "paramdict.h"

namespace spnnruntime
{
    static bool VstrIsFloat(const char vstr[16])
    {
        // look ahead for determine isfloat
        for (int j = 0; j < 16; j++)
        {
            if (vstr[j] == '\0')
                break;

            if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
                return true;
        }

        return false;
    }

    static float VstrToFloat(const char vstr[16])
    {
        double v = 0.0;

        const char *p = vstr;

        // sign
        bool sign = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits before decimal point or exponent
        unsigned int v1 = 0;
        while (isdigit(*p))
        {
            v1 = v1 * 10 + (*p - '0');
            p++;
        }

        v = (double)v1;

        // digits after decimal point
        if (*p == '.')
        {
            p++;

            unsigned int pow10 = 1;
            unsigned int v2 = 0;

            while (isdigit(*p))
            {
                v2 = v2 * 10 + (*p - '0');
                pow10 *= 10;
                p++;
            }

            v += v2 / (double)pow10;
        }

        // exponent
        if (*p == 'e' || *p == 'E')
        {
            p++;

            // sign of exponent
            bool fact = *p != '-';
            if (*p == '+' || *p == '-')
            {
                p++;
            }

            // digits of exponent
            unsigned int expon = 0;
            while (isdigit(*p))
            {
                expon = expon * 10 + (*p - '0');
                p++;
            }

            double scale = 1.0;
            while (expon >= 8)
            {
                scale *= 1e8;
                expon -= 8;
            }
            while (expon > 0)
            {
                scale *= 10.0;
                expon -= 1;
            }

            v = fact ? v * scale : v / scale;
        }

        //     fprintf(stderr, "v = %f\n", v);
        return sign ? (float)v : (float)-v;
    }

    ParamDict::ParamDict()
    {
        params.resize(SPNN_MAX_PARAM_COUNT);
        clear();
    }

    // get type
    int ParamDict::type(int id) const
    {
        return params[id].type;
    }

    // get int
    int ParamDict::get(int id, int def) const
    {
        return params[id].type ? params[id].i : def;
    }

    // get float
    float ParamDict::get(int id, float def) const
    {
        return params[id].type ? params[id].f : def;
    }

    // get array
    Tensor::ptr ParamDict::get(int id, const Tensor::ptr def) const
    {
        return params[id].type ? params[id].v : def;
    }

    // set int
    void ParamDict::set(int id, int i)
    {
        params[id].type = 2;
        params[id].i = i;
    }

    // set float
    void ParamDict::set(int id, float f)
    {
        params[id].type = 3;
        params[id].f = f;
    }

    // set array
    void ParamDict::set(int id, const Tensor::ptr v)
    {
        params[id].type = 4;
        params[id].v = v;
    }

    void ParamDict::clear()
    {
        for (size_t i = 0; i < params.size(); ++i)
        {
            params[i].type = 0;
            params[i].v.reset();
        }
    }

    void ParamDict::loadParam(const DataReader::ptr dr)
    {
        clear();
        // 0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0
        // parse each key=value pair
        int id = 0;
        while (dr->scan("%d=", &id))
        {
            bool isArray = id <= -23300;
            if (isArray)
            {
                id = -id - 23300;
            }
            if (id >= SPNN_MAX_PARAM_COUNT)
            {
                SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "id cannot less than SPNN_MAX_PARAM_COUNT! [id=" << id << ", SPNN_MAX_PARAM_COUNT=" << SPNN_MAX_PARAM_COUNT << "]";
            }
            if (isArray)
            {
                int len = 0;
                bool res = dr->scan("%d", &len);
                if (!res)
                {
                    SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ParamDict read array length failed!";
                }
                params[id].v.reset(new Tensor(len));
                for (int i = 0; i < len; ++i)
                {
                    char vstr[16];
                    res = dr->scan(",%15[^,\n ]", vstr);
                    if (!res)
                    {
                        SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ParamDict read array element failed!";
                    }
                    bool isFloat = VstrIsFloat(vstr);
                    if (isFloat)
                    {
                        float *ptr = *params[id].v;
                        ptr[i] = VstrToFloat(vstr);
                    }
                    else
                    {
                        int *ptr = (int*)(*params[id].v);
                        res = sscanf(vstr, "%d", &ptr[i]);
                        if (!res)
                        {
                            SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ParamDict read array element failed!";
                        }
                    }

                    params[id].type = isFloat ? 6 : 5;
                }
            }
            else
            {
                char vstr[16];
                bool res = dr->scan("%15s", vstr);
                if (!res)
                {
                    SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ParamDict read value failed!";
                }
                bool isFloat = VstrIsFloat(vstr);
                if (isFloat)
                {
                    params[id].f = VstrToFloat(vstr);
                }
                else
                {
                    int ret = sscanf(vstr, "%d", &params[id].i);
                    if (ret == -1)
                    {
                        SPNN_LOG_ERROR(SPNN_ROOT_LOGGER) << "ParamDict read value failed!";
                    }
                }
                params[id].type = isFloat ? 3 : 2;
            }
        }
    }

    void ParamDict::loadParamBin(const DataReader::ptr dr)
    {
    }
} // spnnruntime