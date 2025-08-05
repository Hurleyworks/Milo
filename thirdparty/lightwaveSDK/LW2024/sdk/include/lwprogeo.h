/*
 * LWSDK Header File
 *
 * lwprogeo.h -- LightWave Procedural Geometry
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_PROGEO_H
#define LWSDK_PROGEO_H

#include <lwglobsrv.h>
#include <lwtypes.h>
#include <lwcomring.h>

#define LWGEOOBJECTFUNCS_GLOBAL "ProGeo Funcs 1"
/*
The GeoObject Functions provide access to PeoGeo data
*/
#define geoID       LWID_('G', 'E', 'O', 'O')
#define geoVendorID LWID_(' ', 'P', 'G', 'O')


typedef void* LWGeoObjsID;
typedef void* LWGeoObjID;

typedef struct st_LWGeoObjectFuncs {
    LWGeoObjsID     (*geoObjects)(LWItemID item);
    int             (*getGeoCount)(LWGeoObjsID geos, LWCStringUTF8 name);
    LWGeoObjID      (*getGeoObject)(LWGeoObjsID geos, LWCStringUTF8 name, int num);

    LWGeoObjID      (*createGeoObject)(LWItemID item, LWCStringUTF8 name);
    void            (*destroyGeoObject)(LWGeoObjID geo);
    void            (*clearGeoObject)(LWGeoObjID geo);

    unsigned int    (*numVertices)(LWGeoObjID geoObj);
    LWError         (*position)(LWGeoObjID geoObj, LWPntID pnt, LWFVector pos);
    unsigned int    (*numPolygons)(LWGeoObjID geoObj);
    unsigned int    (*numPolVertices)(LWGeoObjID geoObj, LWPolID pol);
    LWError         (*setPosition)(LWGeoObjID geoObj, LWPntID pnt, const LWFVector pos);
    LWPntID         (*newVertex)(LWGeoObjID geoObj, const LWFVector pos);
    LWPolID         (*newPolygon)(LWGeoObjID geoObj, LWCStringUTF8 surfname, int nvert, LWPntID *vertices);
    void            (*vmapSet)(LWGeoObjID geoObj, LWID type, LWCStringUTF8 name, int dim, LWPntID pid, LWPolID pol, float *vector);
} LWGeoObjectFuncs;
#endif