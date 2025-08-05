/*
 * LWSDK Header File
 *
 * LWTYPES.H -- LightWave Common Types
 *
 * This header contains type declarations common to all aspects of
 * LightWave.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_TYPES_H
#define LWSDK_TYPES_H

#define LWITEM_NULL ((LWItemID)0)

typedef unsigned short   LWDualKey;

typedef void*   LWItemID;
typedef void*   LWImageID;
typedef float   LWBufferValue;
typedef void*   LWPixmapID;
typedef void*   LWTextureID;
typedef void*   NodeEditorID;
typedef void*   ItemInstanceID;
typedef void*   LWUserData;
typedef void*   LWBxDF;
typedef void*   LWBSDF;
typedef void*   LWBSSRDF;
typedef void*   LWBTDFVolume;

typedef enum
{
    LWBTDFVOLUMETYPE_HOMOGENEOUS = 0,
    LWBTDFVOLUMETYPE_HETEROGENEOUS
} LWBTDFVolumeType;

typedef int     LWFrame;
typedef double  LWTime;

typedef float   LWFVector[3];
typedef double  LWDVector[3];
typedef float   LWFVector4[4];
typedef double  LWDVector4[4];
typedef float   LWFMatrix3[3][3];
typedef float   LWFMatrix4[4][4];
typedef double  LWDMatrix3[3][3];
typedef double  LWDMatrix4[4][4];
typedef double  LWDMatrix9[9];

// LightWave C-strings are 0-terminated and ASCII encoded, by default.
// Other encodings are similar but differ in their character encoding, which may be multi-byte per character.
// The ANSI encoding, a legacy construct, refers to encoding that changes based on the LW language choice.
// UTF8 encoding is the preferred character encoding for C-strings when Unicode characters are required.
// All encodings share the ASCII range of character encodings; therefore, ASCII characters will be encoded identically for any encoding.
// Use of "char *" or "const char*" may not refer to C-strings at all.
typedef const char *LWCStringASCII;
typedef const char *LWCStringANSI; // (deprecated)
typedef const char *LWCStringUTF8;
typedef LWCStringASCII LWCString;

// Mutable strings are those that can be changed (not constant)
typedef char *LWMutableCStringASCII;
typedef char *LWMutableCStringANSI; // (deprecated)
typedef char *LWMutableCStringUTF8;
typedef LWMutableCStringASCII LWMutableCString;

typedef void*   LWSurfaceID;

typedef unsigned int     LWID;
#ifndef LWID_
    #define LWID_(a,b,c,d) ((((unsigned int)a)<<24)|(((unsigned int)b)<<16)|(((unsigned int)c)<<8)|((unsigned int)d))
#endif

typedef unsigned long long LWLongID;

#ifndef LWLongID_
    #define LWLongID_(a,b,c,d,e,f,g,h) ((((LWLongID)a)<<56)|(((LWLongID)b)<<48)|(((LWLongID)c)<<40)|(((LWLongID)d)<<32)|(((LWLongID)e)<<24)|(((LWLongID)f)<<16)|(((LWLongID)g)<<8)|((LWLongID)h))
#endif

typedef int              LWCommandCode;

typedef void *           LWChannelID;

/*
 * Persistent instances are just some opaque data object referenced
 * by void pointer.  Errors from handler functions are human-readable
 * strings, where a null string pointer indicates no error.
 */
typedef void *           LWInstance;
typedef LWCStringUTF8    LWError;

#ifndef NULL
    #ifndef __cplusplus
        #define NULL    ((void *) 0)
    #else
        #define NULL     0
    #endif
#endif

/*
 * These are useful to support 64bit architectures of Mac universal binaries
 * where 'long' and pointers are 8bytes (The LP64 programming model)
 */
#define PTR2INT(x) ((int)(size_t)(x))
#define PTR2UINT(x) ((unsigned int)(size_t)(x))
#define INT2PTR(x) ((void*)(size_t)(x))
#define UINT2PTR(x) ((void*)(size_t)(x))

#define LWITEMID2INT(x) PTR2INT(x)
#define LWITEMID2UINT(x) PTR2UINT(x)
#define INT2LWITEMID(x) ((LWItemID)INT2PTR(x))
#define UINT2LWITEMID(x) ((LWItemID)UINT2PTR(x))

typedef LWCStringASCII LWASCIISignature;

#endif
