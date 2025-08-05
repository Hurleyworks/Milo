/*
 * LWSDK Header File
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWMATH_H
#define LWMATH_H

#include <math.h>

#ifndef BIGNUM
    #define BIGNUM          1.0E12   // A number we will hopefully never exceed
#endif

#ifndef PI
    #define PI              (3.141592653589793238462643383279502884197169399375105820974944592307816)
#endif

#ifndef TWOPI
    #define TWOPI           (2.0 * PI)
#endif

#ifndef HALFPI
    #define HALFPI          (0.5 * PI)
#endif

#ifndef FOURPI
#define FOURPI              (4.0 * PI)
#endif

#ifndef INVPI
    #define INVPI           (1.0 / PI)
#endif

#ifndef INVTWOPI
    #define INVTWOPI        (1.0 / TWOPI)
#endif

#ifndef INVFOURPI
#define INVFOURPI           (1.0 / FOURPI)
#endif

#ifndef SQRTPI
    #define SQRTPI          (1.7724538509055160272981674833411)
#endif

#ifndef DEGRAD
    #define DEGRAD          (PI/180.0)
#endif

#ifndef LOG05
    #define LOG05           (-0.693147180559945)
#endif

#ifndef RADIANS
    #define RADIANS(deg)    ((deg)*0.017453292519943295769236907684886)
#endif

#ifndef DEGREES
    #define DEGREES(rad)    ((rad)*57.2957795130823208767981548141052)
#endif

#ifndef DEG2RAD
    #define DEG2RAD(x)      RADIANS(x)
#endif

#ifndef RAD2DEG
    #define RAD2DEG(x)      DEGREES(x)
#endif

/// sets all the vector's components to the specified value (x)
#ifndef VSET
    #define VSET(a,x)       ((a)[0]=(x), (a)[1]=(x), (a)[2]=(x))
#endif

/// sets all the vector's components to 0
#ifndef VCLR
    #define VCLR(a)         VSET(a,0.0)
#endif

/// negate all the vector's components
#ifndef VNEG
    #define VNEG(a)         ((a)[0] = -(a)[0], (a)[1]=-(a)[1], (a)[2]=-(a)[2])
#endif

/// copies one vector to another
#ifndef VCPY
    #define VCPY(a,b)       ((a)[0] =(b)[0], (a)[1] =(b)[1], (a)[2] =(b)[2])
#endif

/// copies one matrix to another
#ifndef MCPY
    #define MCPY(a,b)       ((a)[0] =(b)[0], (a)[1] =(b)[1], (a)[2] =(b)[2], \
                             (a)[3] =(b)[3], (a)[4] =(b)[4], (a)[5] =(b)[5], \
                             (a)[6] =(b)[6], (a)[7] =(b)[7], (a)[8] =(b)[8])
#endif

#ifndef VSCL
    #define VSCL(a,x)       ((a)[0]*= (x),   (a)[1]*= (x),   (a)[2]*= (x))
#endif

/// add one vector to another
#ifndef VADD
    #define VADD(a,b)       ((a)[0]+=(b)[0], (a)[1]+=(b)[1], (a)[2]+=(b)[2])
#endif

/// subtract one vector from another
#ifndef VSUB
    #define VSUB(a,b)       ((a)[0]-=(b)[0], (a)[1]-=(b)[1], (a)[2]-=(b)[2])
#endif

/// adds 'b' scaled by 'x' to 'a'
#ifndef VADDS
    #define VADDS(a,b,x)    ((a)[0]+=(b)[0]*(x), (a)[1]+=(b)[1]*(x), (a)[2]+=(b)[2]*(x))
#endif

/// subtracts 'b' scaled by 'x' from 'a'
#ifndef VSUBS
    #define VSUBS(a,b,x)    ((a)[0]-=(b)[0]*(x), (a)[1]-=(b)[1]*(x), (a)[2]-=(b)[2]*(x))
#endif

/// vector dot product
#ifndef VDOT
    #define VDOT(a,b)       ((a)[0]*(b)[0] + (a)[1]*(b)[1] + (a)[2]*(b)[2])
#endif

#ifndef VLEN_SQUARED
    #define VLEN_SQUARED(a) (VDOT(a,a))
#endif

// returns the magnitude of 'a'
#ifndef VLEN
    #define VLEN(a)         sqrt(VLEN_SQUARED(a))
#endif

#ifndef VCROSS
    #define VCROSS(r,a,b)   ((r)[0] = (a)[1]*(b)[2] - (a)[2]*(b)[1],\
                             (r)[1] = (a)[2]*(b)[0] - (a)[0]*(b)[2],\
                             (r)[2] = (a)[0]*(b)[1] - (a)[1]*(b)[0])
#endif

#ifndef VPERP
    #define VPERP(p,v)      ((p)[0] = (v)[1] - (v)[2],\
                             (p)[1] = (v)[2] - (v)[0],\
                             (p)[2] = (v)[0] - (v)[1])
#endif

#ifndef VMUL
    #define VMUL(a,b)       ((a)[0]*=(b)[0],\
                             (a)[1]*=(b)[1],\
                             (a)[2]*=(b)[2])
#endif

#ifndef VDIVS
    #define VDIVS(a,x)      ((a)[0]/= (x),\
                             (a)[1]/= (x),\
                             (a)[2]/= (x))
#endif

#ifndef VDIV
    #define VDIV(a,b)       ((a)[0]/= (b)[0],\
                             (a)[1]/= (b)[1],\
                             (a)[2]/= (b)[2])
#endif

#ifndef VDIVS1
    #define VDIVS1(r,a,x)   ((r)[0]=(a)[0]/(x),\
                             (r)[1]=(a)[1]/(x),\
                             (r)[2]=(a)[2]/(x))
#endif

#ifndef VCOMB
    #define VCOMB(C,A,a,B,b) ((C)[0] = (A) * (a)[0] + (B) * (b)[0],\
                              (C)[1] = (A) * (a)[1] + (B) * (b)[1],\
                              (C)[2] = (A) * (a)[2] + (B) * (b)[2])
#endif

#ifndef VINV
    #define VINV(a,b)       ((a)[0] = -(b)[0],\
                             (a)[1] = -(b)[1],\
                             (a)[2] = -(b)[2])
#endif

#ifndef VSUBSC
    #define VSUBSC(r,a,x)   ((r)[0]=(a)[0]-(x),\
                             (r)[1]=(a)[1]-(x),\
                             (r)[2]=(a)[2]-(x))
#endif

#ifndef VADDSC
    #define VADDSC(r,a,x)   ((r)[0]=(a)[0]+(x),\
                             (r)[1]=(a)[1]+(x),\
                             (r)[2]=(a)[2]+(x))
#endif

// *********************************************************************************
// the *3 functions assign the result to a third vector (r for result)
// *********************************************************************************
#ifndef VSCL3
    #define VSCL3(r,a,x)    ((r)[0]=(a)[0]*(x),    (r)[1]=(a)[1]*(x),    (r)[2]=(a)[2]*(x))
#endif

#ifndef VADD3
    #define VADD3(r,a,b)    ((r)[0]=(a)[0]+(b)[0], (r)[1]=(a)[1]+(b)[1], (r)[2]=(a)[2]+(b)[2])
#endif

#ifndef VSUB3
    #define VSUB3(r,a,b)    ((r)[0]=(a)[0]-(b)[0], (r)[1]=(a)[1]-(b)[1], (r)[2]=(a)[2]-(b)[2])
#endif

// assigns a + b scaled by x to r
#ifndef VADDS3
    #define VADDS3(r,a,b,x) ((r)[0]=(a)[0]+(b)[0]*(x), (r)[1]=(a)[1]+(b)[1]*(x), (r)[2]=(a)[2]+(b)[2]*(x))
#endif

#ifndef VMUL3
    #define VMUL3(r,a,b)    ((r)[0]=(a)[0]*(b)[0], (r)[1]=(a)[1]*(b)[1], (r)[2]=(a)[2]*(b)[2])
#endif

#ifndef VDIVS3
    #define VDIVS3(r,a,x)   ((r)[0] = (a)[0] / (x),\
                             (r)[1] = (a)[1] / (x),\
                             (r)[2] = (a)[2] / (x))
#endif

#ifndef VDIV3
    #define VDIV3(r,a,b)    ((r)[0] = (a)[0] / (b)[0],\
                             (r)[1] = (a)[1] / (b)[1],\
                             (r)[2] = (a)[2] / (b)[2])
#endif

#ifndef VSUBS3
    #define VSUBS3(r,a,b,x) ((r)[0]=(a)[0]-(b)[0]*(x),\
                             (r)[1]=(a)[1]-(b)[1]*(x),\
                             (r)[2]=(a)[2]-(b)[2]*(x))
#endif

#ifndef VADDS3
    #define VADDS3(r,a,b,x) ((r)[0]=(a)[0]+(b)[0]*(x),\
                             (r)[1]=(a)[1]+(b)[1]*(x),\
                             (r)[2]=(a)[2]+(b)[2]*(x))
#endif

#ifndef VSET3
    #define VSET3(r,x,y,z)  ((r)[0]=(x),\
                             (r)[1]=(y),\
                             (r)[2]=(z))
#endif

#ifndef VTRANSM3
    #define VTRANSM3(r,a,m) ((r)[0] = (a)[0]*(m)[0] + (a)[1]*(m)[3] + (a)[2]*(m)[6],  \
                             (r)[1] = (a)[0]*(m)[1] + (a)[1]*(m)[4] + (a)[2]*(m)[7],  \
                             (r)[2] = (a)[0]*(m)[2] + (a)[1]*(m)[5] + (a)[2]*(m)[8])
#endif

#ifndef ABS

    #define ABS(a)          (((a) < 0) ? (-(a)) : (a))
#endif

#ifndef MAX
    #define MAX(a,b)        ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
    #define MIN(a,b)        ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX3
    #define MAX3(a,b,c)     ((a) > (b) ? ((a) > (c) ? a : c) : ((b) > (c) ? b : c))
#endif

#ifndef MIN3
    #define MIN3(a,b,c)     ((a) < (b) ? ((a) < (c) ? a : c) : ((b) < (c) ? b : c))
#endif

#ifndef CLAMP
    #define CLAMP(a,b,c)    (((a) < (b)) ? (b) : (((a) > (c)) ? (c) : (a)))
#endif

#ifndef APPROX
    #define APPROX(a, b, c) (((a) > ((b) - (c))) && ((a) < ((b) + (c))))
#endif

#ifndef SWAP
    #define SWAP(a,b)       {a^=b; b^=a; a^=b;}
#endif

#ifndef SWAP3
    #define SWAP3(a,b,c)    {(c) = (a); (a) = (b); (b) = (c);}
#endif

#ifndef SIGN
    #define SIGN(a)         (((a) < 0) ? -1 : 1)
#endif

#ifndef SQR
    #define SQR(x)          ((x) * (x))
#endif

#ifndef FLOOR
    #define FLOOR(x)        ((int)(x) - ((x) < 0 && (x) != (int)(x)))
#endif

#ifndef CEIL
    #define CEIL(x)         ((int)(x) + ((x) > 0 && (x) != (int)(x)))
#endif

#ifndef LERP
    #define LERP(t,x0,x1)   ((x0) + ((t) * ((x1) - (x0))))
#endif

#ifndef VLERP3
    #define VLERP3(t,a,b,r) ((r)[0] = LERP((t),(a)[0],(b)[0]),\
                            (r)[1] = LERP((t),(a)[1],(b)[1]),\
                            (r)[2] = LERP((t),(a)[2],(b)[2]))
#endif

#ifndef SMOOTH
    #define SMOOTH(t)       ((t) * (t) * (3 - ((t) + (t))))
#endif

#ifndef SCURVE
    #define SCURVE(t)       SMOOTH(t)
#endif

#ifndef PCURVE
    #define PCURVE(t)       ((t) * (t) * (t) * ((t) * ((t) * 6.0 - 15.0) + 10.0))
#endif

#ifndef BOXSTEP
#define BOXSTEP(a,b,x)      ((a) == (b) ? 0 : CLAMP(((x) - (a)) / ((b) - (a)), 0, 1))
#endif

#ifndef STEP
    #define STEP(a,b)       ((b) >= (a))
#endif

#ifndef PULSE
    #define PULSE(a,b,x)    (STEP((a),(x)) - STEP((b),(x)))
#endif

#ifndef GAMMACORRECT
    #define GAMMACORRECT(gamma, x)  (pow(MAX((x), 0.0 ), 1.0 / (gamma)))
#endif

#ifndef BIAS
    #define BIAS(b,x)       (pow((x), log((b)) / LOG05))
#endif

#ifndef GAIN
    #define GAIN(g,x)       ((x) < 0.5 ? BIAS(1.0 - (g), 2.0 * (x)) / 2.0 : 1.0 - BIAS(1.0 - (g), 2.0 - 2.0 * (x)) / 2.0)
#endif

#ifndef CONTRAST
    #define CONTRAST(contrast, cvalue)\
    {\
        contrast = CLAMP(contrast, -0.999999f, 0.999999f);\
        if ((contrast) > 0) {\
            const float lcon = (contrast) * 0.5f;\
            cvalue = BOXSTEP(lcon, 1 - lcon, cvalue);\
        } else if ((contrast) < 0) {\
            const float fcon = 1 - fabs(contrast);\
            cvalue = BOXSTEP(0, 1, ((cvalue) * fcon) + (0.5f * (1 - fcon)));\
        }\
    }
#endif

#ifndef DOTLL
    #define DOTLL(a,b,c,d,e,f)  ((a) * (d) + (b) * (e) + (c) * (f))
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define LWMATHFUNCS_GLOBAL "LW Math Funcs"


typedef struct st_LWMathFuncs
{
    void (*calcmatrix_rot_9d)  (double m[9],    double h, double p, double b, unsigned int rotation_order);
    void (*calcmatrix_rot_16d) (double m[16],   double h, double p, double b, unsigned int rotation_order);
    void (*calcmatrix_rot_33d) (double m[3][3], double h, double p, double b, unsigned int rotation_order);
    void (*calcmatrix_rot_44d) (double m[4][4], double h, double p, double b, unsigned int rotation_order);

    void (*calcmatrix_rot_9f)  (float m[9],    float h, float p, float b, unsigned int rotation_order);
    void (*calcmatrix_rot_16f) (float m[16],   float h, float p, float b, unsigned int rotation_order);
    void (*calcmatrix_rot_33f) (float m[3][3], float h, float p, float b, unsigned int rotation_order);
    void (*calcmatrix_rot_44f) (float m[4][4], float h, float p, float b, unsigned int rotation_order);


    void (*calcmatrix_rotscale_9d)  (double m[9],    double h, double p, double b, double sx, double sy, double sz, unsigned int rotation_order);
    void (*calcmatrix_rotscale_16d) (double m[16],   double h, double p, double b, double sx, double sy, double sz, unsigned int rotation_order);
    void (*calcmatrix_rotscale_33d) (double m[3][3], double h, double p, double b, double sx, double sy, double sz, unsigned int rotation_order);
    void (*calcmatrix_rotscale_44d) (double m[4][4], double h, double p, double b, double sx, double sy, double sz, unsigned int rotation_order);

    void (*calcmatrix_rotscale_9f)  (float m[9],    float h, float p, float b, float sx, float sy, float sz, unsigned int rotation_order);
    void (*calcmatrix_rotscale_16f) (float m[16],   float h, float p, float b, float sx, float sy, float sz, unsigned int rotation_order);
    void (*calcmatrix_rotscale_33f) (float m[3][3], float h, float p, float b, float sx, float sy, float sz, unsigned int rotation_order);
    void (*calcmatrix_rotscale_44f) (float m[4][4], float h, float p, float b, float sx, float sy, float sz, unsigned int rotation_order);


    void (*calcmatrix_posrotscale_16d) (double m[16],   double tx, double ty, double tz, double h, double p, double b, double sx, double sy, double sz, unsigned int rotation_order);
    void (*calcmatrix_posrotscale_44d) (double m[4][4], double tx, double ty, double tz, double h, double p, double b, double sx, double sy, double sz, unsigned int rotation_order);

    void (*calcmatrix_posrotscale_16f) (float m[16],   float tx, float ty, float tz, float h, float p, float b, float sx, float sy, float sz, unsigned int rotation_order);
    void (*calcmatrix_posrotscale_44f) (float m[4][4], float tx, float ty, float tz, float h, float p, float b, float sx, float sy, float sz, unsigned int rotation_order);


    void (*calcmatrix_posrotscalepivot_16d) (double m[16],   double tx, double ty, double tz, double h, double p, double b, double sx, double sy, double sz, double pivtx, double pivty, double pivtz, double pivh, double pivp, double pivb, unsigned int rotation_order);
    void (*calcmatrix_posrotscalepivot_44d) (double m[4][4], double tx, double ty, double tz, double h, double p, double b, double sx, double sy, double sz, double pivtx, double pivty, double pivtz, double pivh, double pivp, double pivb, unsigned int rotation_order);

    void (*calcmatrix_posrotscalepivot_16f) (float m[16],   float tx, float ty, float tz, float h, float p, float b, float sx, float sy, float sz, float pivtx, float pivty, float pivtz, float pivh, float pivp, float pivb, unsigned int rotation_order);
    void (*calcmatrix_posrotscalepivot_44f) (float m[4][4], float tx, float ty, float tz, float h, float p, float b, float sx, float sy, float sz, float pivtx, float pivty, float pivtz, float pivh, float pivp, float pivb, unsigned int rotation_order);


    int  (*xyz_to_hp_d) (const double xyz[3], double b, double* h, double* p, unsigned int rotation_order);
    int  (*xyz_to_hp_f) (const float  xyz[3], float  b, float*  h, float*  p, unsigned int rotation_order);


    void (*matrix_to_hpb_9d)  (const double m[9],    double* h, double* p, double* b, unsigned int rotation_order);
    void (*matrix_to_hpb_16d) (const double m[16],   double* h, double* p, double* b, unsigned int rotation_order);
    void (*matrix_to_hpb_33d) (const double m[3][3], double* h, double* p, double* b, unsigned int rotation_order);
    void (*matrix_to_hpb_44d) (const double m[4][4], double* h, double* p, double* b, unsigned int rotation_order);

    void (*matrix_to_hpb_9f)  (const float m[9],    float* h, float* p, float* b, unsigned int rotation_order);
    void (*matrix_to_hpb_16f) (const float m[16],   float* h, float* p, float* b, unsigned int rotation_order);
    void (*matrix_to_hpb_33f) (const float m[3][3], float* h, float* p, float* b, unsigned int rotation_order);
    void (*matrix_to_hpb_44f) (const float m[4][4], float* h, float* p, float* b, unsigned int rotation_order);


    void (*conv_rotation_order_d) (const double from_hpb[3], unsigned int from_order, double to_hpb[3], unsigned int to_order);
    void (*conv_rotation_order_f) (const float  from_hpb[3], unsigned int from_order, float  to_hpb[3], unsigned int to_order);


    void (*find_best_rotation_d) (double hpb[3], const double matchtohpb[3], unsigned int rotation_order);
    void (*find_best_rotation_f) (float  hpb[3], const float  matchtohpb[3], unsigned int rotation_order);


    void (*mult_matrix_9d)  (double m[9],    const double left[9],    const double right[9]);
    void (*mult_matrix_16d) (double m[16],   const double left[16],   const double right[16]);
    void (*mult_matrix_33d) (double m[3][3], const double left[3][3], const double right[3][3]);
    void (*mult_matrix_44d) (double m[4][4], const double left[4][4], const double right[4][4]);

    void (*mult_matrix_9f)  (float m[9],    const float left[9],    const float right[9]);
    void (*mult_matrix_16f) (float m[16],   const float left[16],   const float right[16]);
    void (*mult_matrix_33f) (float m[3][3], const float left[3][3], const float right[3][3]);
    void (*mult_matrix_44f) (float m[4][4], const float left[4][4], const float right[4][4]);


    int (*invert_matrix_9d)  (double im[9],    const double m[9]);
    int (*invert_matrix_16d) (double im[16],   const double m[16]);
    int (*invert_matrix_33d) (double im[3][3], const double m[3][3]);
    int (*invert_matrix_44d) (double im[4][4], const double m[4][4]);

    int (*invert_matrix_9f)  (float im[9],    const float m[9]);
    int (*invert_matrix_16f) (float im[16],   const float m[16]);
    int (*invert_matrix_33f) (float im[3][3], const float m[3][3]);
    int (*invert_matrix_44f) (float im[4][4], const float m[4][4]);


    void (*transform_9d)  (double res[3], const double m[9],    const double v[3]);
    void (*transform_16d) (double res[4], const double m[16],   const double v[4]);
    void (*transform_33d) (double res[3], const double m[3][3], const double v[3]);
    void (*transform_44d) (double res[4], const double m[4][4], const double v[4]);

    void (*transform_9f)  (float res[3], const float m[9],    const float v[3]);
    void (*transform_16f) (float res[4], const float m[16],   const float v[4]);
    void (*transform_33f) (float res[3], const float m[3][3], const float v[3]);
    void (*transform_44f) (float res[4], const float m[4][4], const float v[4]);
} LWMathFuncs;

#ifdef __cplusplus
}
#endif

#endif