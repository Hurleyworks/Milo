/*
 * LWSDK Header File
 *
 * LWSURFED.H -- LightWave Surface Editor
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_SURFED_H
#define LWSDK_SURFED_H

#include <lwsurf.h>

#define LWSURFEDFUNCS_GLOBAL    "SurfaceEditor Functions 2"

#define LWSURFEDFUNCS_PRIMARYINDEX -1

typedef struct st_LWSurfEdFuncs {
    void            (*open)(int);
    void            (*close)(void);
    int             (*isOpen)(void);
    void            (*setSurface)(LWSurfaceID);
    void            (*setPosition)(int  x, int  y);
    void            (*getPosition)(int  *x, int *y,int *w,int *h);
    void            (*addLib)(LWSurfLibID lib);
    LWSurfaceID     (*getSelection)(int index);
} LWSurfEdFuncs;

#endif