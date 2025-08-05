/*
* LWSDK Header File
*
* lwattrdefine.H -- LightWave Attributes Define Functions
*
*Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
*
*This file contains confidential and proprietary information of LightWave Digital, Ltd.,
*and is subject to the terms of the LightWave End User License Agreement (EULA).
*/

#ifndef LWCPP_LWATTRDEFINE_H
#define LWCPP_LWATTRDEFINE_H

#include "lwattrglobal.h"

namespace lwsdk
{
    /**
     *      @brief LWAttrStringDefine
     *
     *      @short Contains the string definition class.
     */

    class LWAttrStringDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrString *m_data;

            LWAttrStringDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachString( attrbase, name ) )
            {
            }

        public:
            LWAttrStringDefine( LWAttributeID attrbase, const char *name, const char *value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createString( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const char *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const char *value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrStringAttach
     *
     *      @short Contains the string attach class.
     */

    class LWAttrStringAttach : public LWAttrStringDefine
    {
        public:
            LWAttrStringAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrStringDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrIntDefine
     *
     *      @short Contains the int definition class.
     */

    class LWAttrIntDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrInt    *m_data;

            LWAttrIntDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachInt( attrbase, name ) )
            {
            }

        public:
            LWAttrIntDefine( LWAttributeID attrbase, const char *name, const int value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createInt( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            int value()
            {
                return m_data->value( m_data );
            }

            int setValue( const int value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrIntAttach
     *
     *      @short Contains the int attach class.
     */

    class LWAttrIntAttach : public LWAttrIntDefine
    {
        public:
            LWAttrIntAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrIntDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrGradientDefine
     *
     *      @short Contains the gradient definition class.
     */

    class LWAttrGradientDefine
    {
        protected:
            LWAttributeID   m_attrbase;
            LWAttrGradient *m_data;

            LWAttrGradientDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachGradient( attrbase, name ) )
            {
            }

        public:
            LWAttrGradientDefine( LWAttributeID attrbase, const char *name, const char *tag, const unsigned int numberChannels ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createGradient( attrbase, name, tag, numberChannels ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            LWGradientRef value()
            {
                return m_data->value( m_data );
            }

            int setValue( const LWGradientRef value )
            {
                return m_data->setValue( m_data, value );
            }

            int setGradientState( int state )
            {
                return m_data->setGradientState( m_data, state );
            }

            int getGradientState()
            {
                return m_data->getGradientState( m_data );
            }

            const char *gradientType()
            {
                return m_data->gradientType( m_data );
            }
    };

    /**
     *      @brief LWAttrGradientAttach
     *
     *      @short Contains the gradient attach class.
     */

    class LWAttrGradientAttach : public LWAttrGradientDefine
    {
        public:
            LWAttrGradientAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrGradientDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrReferenceDefine
     *
     *      @short Contains the reference definition class.
     */

    class LWAttrReferenceDefine
    {
        protected:
            LWAttributeID    m_attrbase;
            LWAttrReference *m_data;

            LWAttrReferenceDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachReference( attrbase, name ) )
            {
            }

        public:
            LWAttrReferenceDefine( LWAttributeID attrbase, const char *name, const void *value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createReference( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const void *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const void *value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrReferenceAttach
     *
     *      @short Contains the reference attach class.
     */

    class LWAttrReferenceAttach : public LWAttrReferenceDefine
    {
        public:
            LWAttrReferenceAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrReferenceDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrImageDefine
     *
     *      @short Contains the image definition class.
     */

    class LWAttrImageDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrImage  *m_data;

            LWAttrImageDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachImage( attrbase, name ) )
            {
            }

        public:
            LWAttrImageDefine( LWAttributeID attrbase, const char *name, const LWImageID value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createImage( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            LWImageID value()
            {
                return m_data->value( m_data );
            }

            int setValue( const LWImageID value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrImageAttach
     *
     *      @short Contains the image attach class.
     */

    class LWAttrImageAttach : public LWAttrImageDefine
    {
        public:
            LWAttrImageAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrImageDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrFileDefine
     *
     *      @short Contains the file name definition class.
     */

    class LWAttrFileDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrFile   *m_data;

            LWAttrFileDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachFile( attrbase, name ) )
            {
            }

        public:
            LWAttrFileDefine( LWAttributeID attrbase, const char *name, const char *value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createFile( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const char *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const char *value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrFileAttach
     *
     *      @short Contains the file name attach class.
     */

    class LWAttrFileAttach : public LWAttrFileDefine
    {
        public:
            LWAttrFileAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrFileDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrFloatDefine
     *
     *      @short Contains the float definition class.
     */

    class LWAttrFloatDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrFloat  *m_data;

            LWAttrFloatDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachFloat( attrbase, name ) )
            {
            }

        public:
            LWAttrFloatDefine( LWAttributeID attrbase, const char *name, const float value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createFloat( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            float value()
            {
                return m_data->value( m_data );
            }

            int setValue( const float value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrFloatAttach
     *
     *      @short Contains the float attach class.
     */

    class LWAttrFloatAttach : public LWAttrFloatDefine
    {
        public:
            LWAttrFloatAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrFloatDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrFloat3Define
     *
     *      @short Contains the float 3 definition class.
     */

    class LWAttrFloat3Define
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrFloat3 *m_data;

            LWAttrFloat3Define( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachFloat3( attrbase, name ) )
            {
            }

        public:
            LWAttrFloat3Define( LWAttributeID attrbase, const char *name, const float value[ 3 ] ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createFloat3( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const float *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const float value[ 3 ] )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrFloat3Attach
     *
     *      @short Contains the float 3 attach class.
     */

    class LWAttrFloat3Attach : public LWAttrFloat3Define
    {
        public:
            LWAttrFloat3Attach( LWAttributeID attrbase, const char *name ) :
                 LWAttrFloat3Define( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrFloat4Define
     *
     *      @short Contains the float 4 definition class.
     */

    class LWAttrFloat4Define
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrFloat4 *m_data;

            LWAttrFloat4Define( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachFloat4( attrbase, name ) )
            {
            }

        public:
            LWAttrFloat4Define( LWAttributeID attrbase, const char *name, const float value[ 4 ] ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createFloat4( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const float *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const float value[ 4 ] )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrFloat4Attach
     *
     *      @short Contains the float 4 attach class.
     */

    class LWAttrFloat4Attach : public LWAttrFloat4Define
    {
        public:
            LWAttrFloat4Attach( LWAttributeID attrbase, const char *name ) :
                 LWAttrFloat4Define( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrDoubleDefine
     *
     *      @short Contains the double definition class.
     */

    class LWAttrDoubleDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrDouble *m_data;

            LWAttrDoubleDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachDouble( attrbase, name ) )
            {
            }

        public:
            LWAttrDoubleDefine( LWAttributeID attrbase, const char *name, const double value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createDouble( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            double value()
            {
                return m_data->value( m_data );
            }

            int setValue( const double value )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrDoubleAttach
     *
     *      @short Contains the double attach class.
     */

    class LWAttrDoubleAttach : public LWAttrDoubleDefine
    {
        public:
            LWAttrDoubleAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrDoubleDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrDouble3Define
     *
     *      @short Contains the double 3 definition class.
     */

    class LWAttrDouble3Define
    {
        protected:
            LWAttributeID  m_attrbase;
            LWAttrDouble3 *m_data;

            LWAttrDouble3Define( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachDouble3( attrbase, name ) )
            {
            }

        public:
            LWAttrDouble3Define( LWAttributeID attrbase, const char *name, const double value[ 3 ] ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createDouble3( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const double *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const double value[ 3 ] )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrDouble3Attach
     *
     *      @short Contains the double 3 attach class.
     */

    class LWAttrDouble3Attach : public LWAttrDouble3Define
    {
        public:
            LWAttrDouble3Attach( LWAttributeID attrbase, const char *name ) :
                 LWAttrDouble3Define( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrDouble4Define
     *
     *      @short Contains the double 4 definition class.
     */

    class LWAttrDouble4Define
    {
        protected:
            LWAttributeID  m_attrbase;
            LWAttrDouble4 *m_data;

            LWAttrDouble4Define( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachDouble4( attrbase, name ) )
            {
            }

        public:
            LWAttrDouble4Define( LWAttributeID attrbase, const char *name, const double value[ 4 ] ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createDouble4( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const double *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const double value[ 4 ] )
            {
                return m_data->setValue( m_data, value );
            }
    };

    /**
     *      @brief LWAttrDouble4Attach
     *
     *      @short Contains the double 4 attach class.
     */

    class LWAttrDouble4Attach : public LWAttrDouble4Define
    {
        public:
            LWAttrDouble4Attach( LWAttributeID attrbase, const char *name ) :
                 LWAttrDouble4Define( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrVParamDefine
     *
     *      @short Contains the vparam definition class.
     */

    class LWAttrVParamDefine
    {
        protected:
            LWAttributeID m_attrbase;
            LWAttrVParam *m_data;

            LWAttrVParamDefine( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachVParam( attrbase, name ) )
            {
            }

        public:
            LWAttrVParamDefine( LWAttributeID attrbase, const char *name, const float value ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createVParam( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            float value()
            {
                return m_data->value( m_data );
            }

            int setValue( const float value )
            {
                return m_data->setValue( m_data, value );
            }

            float evaluate( LWTime time )
            {
                return m_data->evaluate( m_data, time );
            }

            LWEnvelopeID createEnvelope( const char *type )
            {
                return m_data->createEnvelope( m_data, type );
            }

            void destroyEnvelope()
            {
                m_data->destroyEnvelope( m_data );
            }

            LWEnvelopeID envelope()
            {
                return m_data->envelope( m_data );
            }

            void setChannelGroup( const char *namedMemory, LWChanGroupID group )
            {
                m_data->setChannelGroup( m_data, namedMemory, group );
            }

            int setEnvelopeState( int enable )
            {
                return m_data->setEnvelopeState( m_data, enable );
            }

            int getEnvelopeState()
            {
                return m_data->getEnvelopeState( m_data );
            }

            const char *envelopeType()
            {
                return m_data->envelopeType( m_data );
            }

            LWTextureID createTexture()
            {
                return m_data->createTexture( m_data );
            }

            void destroyTexture()
            {
                m_data->destroyTexture( m_data );
            }

            LWTextureID texture()
            {
                return m_data->texture( m_data );
            }

            void setTextureContext( const char *namedMemory, LWTxtrContextID context, void *userdata )
            {
                m_data->setTextureContext( m_data, namedMemory, context, userdata );
            }

            int setTextureState( int state )
            {
                return m_data->setTextureState( m_data, state );
            }

            int getTextureState()
            {
                return m_data->getTextureState( m_data );
            }
    };

    /**
     *      @brief LWAttrVParamAttach
     *
     *      @short Contains the vparam attach class.
     */

    class LWAttrVParamAttach : public LWAttrVParamDefine
    {
        public:
            LWAttrVParamAttach( LWAttributeID attrbase, const char *name ) :
                 LWAttrVParamDefine( attrbase, name )
            {
            }
    };

    /**
     *      @brief LWAttrVParam3Define
     *
     *      @short Contains the vparam 3 definition class.
     */

    class LWAttrVParam3Define
    {
        protected:
            LWAttributeID  m_attrbase;
            LWAttrVParam3 *m_data;

            LWAttrVParam3Define( LWAttributeID attrbase, const char *name ) :
                m_attrbase( attrbase ),
                m_data( attrattachfunc->attachVParam3( attrbase, name ) )
            {
            }

        public:
            LWAttrVParam3Define( LWAttributeID attrbase, const char *name, const double value[ 3 ] ) :
                m_attrbase( attrbase ),
                m_data( attrdefinefunc->createVParam3( attrbase, name, value ) )
            {
            }

            int destroy()
            {
                int result = m_data->destroy( m_data );
                m_data = nullptr;
                return result;
            }

            LWAttributeID baseAttr()
            {
                return m_data->baseAttr( m_data );
            }

            LWNameID nameID()
            {
                return m_data->nameID( m_data );
            }

            void setFlags( unsigned int flags, unsigned int mask )
            {
                return m_data->setFlags( m_data, flags, mask );
            }

            unsigned int getFlags()
            {
                return m_data->getFlags( m_data );
            }

            const double *value()
            {
                return m_data->value( m_data );
            }

            int setValue( const double value[ 3 ] )
            {
                return m_data->setValue( m_data, value );
            }

            const float *evaluate( LWTime time )
            {
                return m_data->evaluate( m_data, time );
            }

            const LWEnvelopeID *createEnvelopes( const char *type )
            {
                return m_data->createEnvelopes( m_data, type );
            }

            void destroyEnvelopes()
            {
                m_data->destroyEnvelopes( m_data );
            }

            const LWEnvelopeID *envelopes()
            {
                return m_data->envelopes( m_data );
            }

            void setChannelGroup( const char *namedMemory, LWChanGroupID group )
            {
                m_data->setChannelGroup( m_data, namedMemory, group );
            }

            int setEnvelopeStates( int enable[ 3 ] )
            {
                return m_data->setEnvelopeStates( m_data, enable );
            }

            int *getEnvelopeStates()
            {
                return m_data->getEnvelopeStates( m_data );
            }

            const char *envelopesType()
            {
                return m_data->envelopesType( m_data );
            }

            LWTextureID createTexture()
            {
                return m_data->createTexture( m_data );
            }

            void destroyTexture()
            {
                m_data->destroyTexture( m_data );
            }

            LWTextureID texture()
            {
                return m_data->texture( m_data );
            }

            void setTextureContext( const char *namedMemory, LWTxtrContextID context, void *userdata )
            {
                m_data->setTextureContext( m_data, namedMemory, context, userdata );
            }

            int setTextureState( int state )
            {
                return m_data->setTextureState( m_data, state );
            }

            int getTextureState()
            {
                return m_data->getTextureState( m_data );
            }
    };

    /**
     *      @brief LWAttrVParam3Attach
     *
     *      @short Contains the vparam 3 attach class.
     */

    class LWAttrVParam3Attach : public LWAttrVParam3Define
    {
        public:
            LWAttrVParam3Attach( LWAttributeID attrbase, const char *name ) :
                 LWAttrVParam3Define( attrbase, name )
            {
            }
    };
}

#endif