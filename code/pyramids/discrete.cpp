#include "discrete.h"

const float kernel::discrete::delta::p_v[1] = {1.f};

const float kernel::discrete::ifir_condat0::p_v[3] = {-1.f/24.f, 13.f/12.f, 
    -1.f/24.f};

const float kernel::discrete::ifir_condat1::p_v[3] = {1.f/12.f, 5.f/6.f, 
    1.f/12.f};

const float kernel::discrete::ifir_condat2::p_v[5] = {-7.f/1920.f, 67.f/480.f, 
    233.f/320.f, 67.f/480.f, -7.f/1920.f};

const float kernel::discrete::ifir_condat3::p_v[5] = {-1.f/720.f, 31.f/180.f, 
    79.f/120.f, 31.f/180.f, -1.f/720.f};

const float kernel::discrete::ifir_condat_omoms3::p_v[5] = {1.f/1680.f, 
    79.f/420.f, 523.f/840.f, 79.f/420.f, 1.f/1680.f};

const float kernel::discrete::fir_dalai1::p_v[5] = {7.f/720.f, -11.f/90.f, 
    49.f/40.f, -11.f/90.f, 7.f/720.f};

const float kernel::discrete::fir_dalai2::p_v[5] = {37.f/1920.f, -97.f/480.f, 
    437.f/320.f, -97.f/480.f, 37.f/1920.f};

const float kernel::discrete::fir_dalai3::p_v[7] = {-41.f/7560.f, 
    311.f/5040.f, -919.f/2520.f, 12223.f/7560.f, -919.f/2520.f, 
    311.f/5040.f, -41.f/7560.f};

const float kernel::discrete::a_hat::p_v[3] = {1.f/6.f, 2.f/3.f, 
    1.f/6.f}; 

const float kernel::discrete::a_bspline2::p_v[5] = {1.f/120.f, 13.f/60.f,
    11.f/20.f, 13.f/60.f, 1.f/120.f}; 

const float kernel::discrete::a_bspline3::p_v[7] = {1.f/5040.f, 1.f/42.f,
    397.f/1680.f, 151.f/315.f, 397.f/1680.f, 1.f/42.f, 1.f/5040.f};

const float kernel::discrete::a_bspline4::p_v[9] = {1.f/362880.f, 
    251.f/181440.f, 913.f/22680.f, 44117.f/181440.f, 15619.f/36288.f, 
    44117.f/181440.f, 913.f/22680.f, 251.f/181440.f, 1.f/362880.f};

const float kernel::discrete::a_bspline5::p_v[11] = {
    1.f/39916800.f, 509.f/9979200.f, 50879.f/13305600.f, 1093.f/19800.f, 
    1623019.f/6652800.f, 655177.f/1663200.f, 1623019.f/6652800.f, 
    1093.f/19800.f, 50879.f/13305600.f, 509.f/9979200.f, 1.f/39916800.f};

const float kernel::discrete::a_omoms2::p_v[5] = {17.f/1200.f, 17.f/75.f,
    311.f/600.f, 17.f/75.f, 17.f/1200.f};

const float kernel::discrete::a_omoms3::p_v[7] = {73.f/105840.f, 1.f/30.f,
    2839.f/11760.f, 2971.f/6615.f, 2839.f/11760.f, 1.f/30.f, 73.f/105840.f};

const float kernel::discrete::a_omoms4::p_v[9] = {557.f/25401600.f, 
    13567.f/4762800.f, 45817.f/907200.f, 18463.f/75600.f, 88139.f/217728.f,
    18463.f/75600.f, 45817.f/907200.f, 13567.f/4762800.f, 557.f/25401600.f};

const float kernel::discrete::a_omoms5::p_v[11] = {431.f/878169600.f, 
    1429.f/8781696.f, 73273.f/11708928.f, 7181.f/110880.f, 7101889.f/29272320.f,
    13626317.f/36590400.f, 7101889.f/29272320.f, 7181.f/110880.f, 
    73273.f/11708928.f, 1429.f/8781696.f, 431.f/878169600.f};

const float kernel::discrete::a_mitchell_netravali::p_v[7] = {1.f/2160.f, 
    -7.f/405.f, 1141.f/6480.f, 92.f/135.f, 1141.f/6480.f, -7.f/405.f, 
    1.f/2160.f};

const float kernel::discrete::a_keys4::p_v[7] = {1.f/560.f, -1.f/28.f, 
    71.f/560.f, 57.f/70.f, 71.f/560.f, -1.f/28.f, 1.f/560.f};

const float kernel::discrete::a_crt::p_v[11] = {-0.0000237287f, 0.0000257689f,
    0.00264125f, 0.0445654f, 0.241124f, 0.423222f, 0.241124f, 0.0445654f, 
    0.00264125f, 0.0000257689f, -0.0000237287f};

const float kernel::discrete::a_bspline3_box::p_v[5] = {1.f/384.f, 19.f/96.f, 
    115.f/192.f, 19.f/96.f, 1.f/384.f};

const float kernel::discrete::a_omoms3_box::p_v[5] = {5.f/896.f, 47.f/224.f, 
    255.f/448.f, 47.f/224.f, 5.f/896.f};

const float kernel::discrete::a_keys4_box::p_v[5] = {-5.f/384.f, 3.f/32.f, 
    161.f/192.f, 3.f/32.f, -5.f/384.f};

const float kernel::discrete::a_mitchell_netravali_box::p_v[5] = {-1.f/128.f, 
    37.f/288.f, 437.f/576.f, 37.f/288.f, -1.f/128.f};

const float kernel::discrete::a_bspline5_box::p_v[7] = {1.f/46080.f, 
    361.f/23040.f, 10543.f/46080.f, 5887.f/11520.f, 10543.f/46080.f, 
    361.f/23040.f, 1.f/46080.f};

const float kernel::discrete::a_omoms5_box::p_v[7] = {59.f/506880.f, 
    5459.f/253440.f, 118997.f/506880.f, 61733.f/126720.f, 118997.f/506880.f, 
    5459.f/253440.f, 59.f/506880.f};

const float kernel::discrete::fir_blu35::p_v[9] = {1.875e-6f, 0.012285f, 
    0.621773f, 4.65992f, 8.76565f, 4.65992f, 0.621773f, 0.012285f, 1.875e-6f};

const float kernel::discrete::ifir_blu35::p_v[11] = {0.00001f, 0.00196f, 
    0.10541f, 1.20608f, 4.67858f, 7.36952f, 4.67858f, 1.20608f, 0.10541f, 
    0.00196f, 0.00001f};

const float kernel::discrete::mccool::p_v[5] = {1.f/36.f, 2.f/9.f, 1.f/2.f, 
    2.f/9.f, 1.f/36.f};
