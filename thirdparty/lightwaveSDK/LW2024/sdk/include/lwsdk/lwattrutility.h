/*
* LWSDK Header File
*
* lwattrutility.H -- LightWave Attributes Utility Functions
*
*Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
*
*This file contains confidential and proprietary information of LightWave Digital, Ltd.,
*and is subject to the terms of the LightWave End User License Agreement (EULA).
*/

#ifndef LWCPP_LWATTRUTILITY_H
#define LWCPP_LWATTRUTILITY_H

#include "lwattrglobal.h"
#include <lwsdk/Globals.h>

namespace lwsdk
{
    /**
     *      @brief LWAttrUtilityClass
     *
     *      @short This is the cpp wrapper for the attribute utility.
     */

    class LWAttrUtilityClass : public LWAttrUtility
    {
        public:

        LWAttrUtilityClass()
        {
        }

        LWAttributeID create( size_t maxEnum )
        {
            return attrutilfunc->LWAttrUtility::create( maxEnum );
        }

        int destroy( LWAttributeID attrbase )
        {
            return attrutilfunc->LWAttrUtility::destroy( attrbase );
        }

        LWError copy( LWAttributeID dst, LWAttributeID src )
        {
             return attrutilfunc->LWAttrUtility::copy( dst, src );
        }

        int load( LWAttributeID attrbase, const LWLoadState *load )
        {
            return attrutilfunc->LWAttrUtility::load( attrbase, load );
        }

        int save( LWAttributeID attrbase, const LWSaveState *save )
        {
            return attrutilfunc->LWAttrUtility::save( attrbase, save );
        }

        int setEvent( LWAttributeID attrbase, LWAttrEventFunc function, void *userData )
        {
            return attrutilfunc->LWAttrUtility::setEvent( attrbase, function, userData );
        }

        void *namedMemory( LWAttributeID attrbase, const char *name, size_t size )
        {
            return attrutilfunc->LWAttrUtility::namedMemory( attrbase, name, size );
        }

        LWNameID nextName( LWAttributeID attrbase, LWGroupID group, LWNameID name )
        {
            return attrutilfunc->LWAttrUtility::nextName( attrbase, group, name );
        }

        const char *getNameFromId( LWAttributeID attrbase, LWNameID name )
        {
            return attrutilfunc->LWAttrUtility::getNameFromId( attrbase, name );
        }

        LWLongID type( LWAttributeID attrbase, LWNameID name )
        {
            return attrutilfunc->LWAttrUtility::type( attrbase, name );
        }
    };
}

#endif