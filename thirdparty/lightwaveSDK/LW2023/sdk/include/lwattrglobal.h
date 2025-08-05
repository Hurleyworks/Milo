/*
* LWSDK Header File
*
* lwattrglobal.h -- LightWave Attributes Global Functions
*
*Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
*
*This file contains confidential and proprietary information of LightWave Digital, Ltd.,
*and is subject to the terms of the LightWave End User License Agreement (EULA).
*/

#ifndef LWSDK_ATTRGLOBAL_H
#define LWSDK_ATTRGLOBAL_H

#include <lwtypes.h>
#include <lwbase.h>
#include <lwenvel.h>
#include <lwgradient.h>
#include <lwtxtr.h>

#define LWATTRUTILITYFUNCS_GLOBAL "LW Attr Utility Funcs"
#define LWATTRDEFINEFUNCS_GLOBAL  "LW Attr Define Funcs"
#define LWATTRATTACHFUNCS_GLOBAL  "LW Attr Attach Funcs"

/*!< These are the ID string for the sdk wrapper. */

typedef struct st_LWAttrString *    LWAttrStringID;
typedef struct st_LWAttrInt *       LWAttrIntID;
typedef struct st_LWAttrGradient *  LWAttrGradientID;
typedef struct st_LWAttrReference * LWAttrReferenceID;
typedef struct st_LWAttrImage *     LWAttrImageID;
typedef struct st_LWAttrFile *      LWAttrFileID;
typedef struct st_LWAttrFloat *     LWAttrFloatID;
typedef struct st_LWAttrFloat3 *    LWAttrFloat3ID;
typedef struct st_LWAttrFloat4 *    LWAttrFloat4ID;
typedef struct st_LWAttrDouble *    LWAttrDoubleID;
typedef struct st_LWAttrDouble3 *   LWAttrDouble3ID;
typedef struct st_LWAttrDouble4 *   LWAttrDouble4ID;
typedef struct st_LWAttrVParam *    LWAttrVParamID;
typedef struct st_LWAttrVParam3 *   LWAttrVParam3ID;

 /*!< Identifies an Attribute uniquely in conjunction with a LWAttributeID. */

typedef size_t LWNameID;

/*!< 0 is defined as invalid Attribute id. */

#define LW_INVALID_ATTR_ID 0

/*!< The ID of a group of Attributes within a container. */

typedef struct st_lwgroup *LWGroupID;

/*!< Generic attributes change events. */

typedef enum {
    LWATTR_PRE_CHANGE,
    LWATTR_POST_CHANGE,
    LWATTR_EVENT_DESTROY,
    LWATTR_PRE_LOAD,
    LWATTR_POST_LOAD
} LWAttributeEvent;

/* VParam attributes change events. */

typedef enum {
    LWATTR_ENV_CREATE = 256,
    LWATTR_ENV_DESTROY,
    LWATTR_ENV_UPDATE,
    LWATTR_ENV_TRACK,
    LWATTR_TXTR_CREATE,
    LWATTR_TXTR_DESTROY,
    LWATTR_TXTR_UPDATE,
    LWATTR_TXTR_TRACK,
    LWATTR_TXTR_AUTOSIZE
} LWAttributeVParamEvent;

/* Gradient attributes change events. */

typedef enum {
    LWATTR_GRADIENT_CREATE = 512,
    LWATTR_GRADIENT_DESTROY,
    LWATTR_GRADIENT_UPDATE,
    LWATTR_GRADIENT_TRACK
} LWAttributeGradientEvent;

typedef struct st_LWAttributeEventData {
    LWNameID nameID;
    size_t   arrayIndex;
    void    *data;
} LWAttributeEventData;

typedef int (*LWAttrEventFunc)( void *userData, LWAttributeID attrs, int event, LWAttributeEventData *eventData );

/* Attribute flags ( via getFlags/setFlags ) ( bits 0-29 only ) */

#define LWATTRIBUTE_FLAG_USERHIDDEN_MASK    (1<<0)  /*!< User hidden attributes are not seen by the user under normal circumstances */
#define LWATTRIBUTE_FLAG_VOLATILE_MASK      (1<<1)  /*!< Volatile attributes are not saved between sessions                         */
#define LWATTRIBUTE_FLAG_SERVERDEFINED_MASK (1<<2)  /*!< attributes are defined by a server's attribute handler                     */

#define LWATTRIBUTE_FLAG_ALL_MASK        0x3fffffff /*!< these bits are valid                                                       */
#define LWATTRIBUTE_FLAG_RESERVED_MASK   0xc0000000 /*!< these bits are reserved for future expansion                               */

enum LWATTR_STATE {
    LWATTR_STATE_NONE,      /*!< Can not allow envelope / textures, no E / T button. */
    LWATTR_STATE_ALLOWED,   /*!< Show button. */
    LWATTR_STATE_ALLOCATED  /*!< Show high lighted button.*/
};

/*!< Contains the wrapper for string. */

typedef struct st_LWAttrString {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrStringID );
    LWAttributeID   (*baseAttr)( LWAttrStringID );
    LWNameID        (*nameID)  ( LWAttrStringID );
    void            (*setFlags)( LWAttrStringID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrStringID );
    LWCStringUTF8   (*value)   ( LWAttrStringID );
    int             (*setValue)( LWAttrStringID, LWCStringUTF8 );
} LWAttrString;

/*!< Contains the wrapper for int. */

typedef struct st_LWAttrInt {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrIntID );
    LWAttributeID   (*baseAttr)( LWAttrIntID );
    LWNameID        (*nameID)  ( LWAttrIntID );
    void            (*setFlags)( LWAttrIntID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrIntID );
    int             (*value)   ( LWAttrIntID );
    int             (*setValue)( LWAttrIntID, const int );
} LWAttrInt;

/*!< Contains the wrapper for gradient. */

typedef struct st_LWAttrGradient {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy)         ( LWAttrGradientID );
    LWAttributeID   (*baseAttr)        ( LWAttrGradientID );
    LWNameID        (*nameID)          ( LWAttrGradientID );
    void            (*setFlags)        ( LWAttrGradientID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)        ( LWAttrGradientID );
    LWGradientRef   (*value)           ( LWAttrGradientID );
    int             (*setValue)        ( LWAttrGradientID, const LWGradientRef );
    int             (*setGradientState)( LWAttrGradientID, int state );
    int             (*getGradientState)( LWAttrGradientID );
    LWCStringASCII  (*gradientType)    ( LWAttrGradientID );
} LWAttrGradient;

/*!< Contains the wrapper for reference. */

typedef struct st_LWAttrReference {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrReferenceID );
    LWAttributeID   (*baseAttr)( LWAttrReferenceID );
    LWNameID        (*nameID)  ( LWAttrReferenceID );
    void            (*setFlags)( LWAttrReferenceID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrReferenceID );
    const void *    (*value)   ( LWAttrReferenceID );
    int             (*setValue)( LWAttrReferenceID, const void * );
} LWAttrReference;

/*!< Contains the wrapper for image id. */

typedef struct st_LWAttrImage {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrImageID );
    LWAttributeID   (*baseAttr)( LWAttrImageID );
    LWNameID        (*nameID)  ( LWAttrImageID );
    void            (*setFlags)( LWAttrImageID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrImageID );
    LWImageID       (*value)   ( LWAttrImageID );
    int             (*setValue)( LWAttrImageID, const LWImageID );
} LWAttrImage;

/*!< Contains the wrapper for file. */

typedef struct st_LWAttrFile {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrFileID );
    LWAttributeID   (*baseAttr)( LWAttrFileID );
    LWNameID        (*nameID)  ( LWAttrFileID );
    void            (*setFlags)( LWAttrFileID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrFileID );
    LWCStringUTF8   (*value)   ( LWAttrFileID );
    int             (*setValue)( LWAttrFileID, LWCStringUTF8 );
} LWAttrFile;

/*!< Contains the wrapper for float. */

typedef struct st_LWAttrFloat {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrFloatID );
    LWAttributeID   (*baseAttr)( LWAttrFloatID );
    LWNameID        (*nameID)  ( LWAttrFloatID );
    void            (*setFlags)( LWAttrFloatID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrFloatID );
    float           (*value)   ( LWAttrFloatID );
    int             (*setValue)( LWAttrFloatID, const float );
} LWAttrFloat;

/*!< Contains the wrapper for float[ 3 ]. */

typedef struct st_LWAttrFloat3 {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrFloat3ID );
    LWAttributeID   (*baseAttr)( LWAttrFloat3ID );
    LWNameID        (*nameID)  ( LWAttrFloat3ID );
    void            (*setFlags)( LWAttrFloat3ID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrFloat3ID );
    const float *   (*value)   ( LWAttrFloat3ID );
    int             (*setValue)( LWAttrFloat3ID, const float value[ 3 ] );
} LWAttrFloat3;

/*!< Contains the wrapper for float[ 4 ]. */

typedef struct st_LWAttrFloat4 {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrFloat4ID );
    LWAttributeID   (*baseAttr)( LWAttrFloat4ID );
    LWNameID        (*nameID)  ( LWAttrFloat4ID );
    void            (*setFlags)( LWAttrFloat4ID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrFloat4ID );
    const float *   (*value)   ( LWAttrFloat4ID );
    int             (*setValue)( LWAttrFloat4ID, const float value[ 4 ] );
} LWAttrFloat4;

/*!< Contains the wrapper for double. */

typedef struct st_LWAttrDouble {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrDoubleID );
    LWAttributeID   (*baseAttr)( LWAttrDoubleID );
    LWNameID        (*nameID)  ( LWAttrDoubleID );
    void            (*setFlags)( LWAttrDoubleID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrDoubleID );
    double          (*value)   ( LWAttrDoubleID );
    int             (*setValue)( LWAttrDoubleID, const double );
} LWAttrDouble;

/*!< Contains the wrapper for double[ 3 ]. */

typedef struct st_LWAttrDouble3 {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrDouble3ID );
    LWAttributeID   (*baseAttr)( LWAttrDouble3ID );
    LWNameID        (*nameID)  ( LWAttrDouble3ID );
    void            (*setFlags)( LWAttrDouble3ID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrDouble3ID );
    const double *  (*value)   ( LWAttrDouble3ID );
    int             (*setValue)( LWAttrDouble3ID, const double value[ 3 ] );
} LWAttrDouble3;

/*!< Contains the wrapper for double[ 4 ]. */

typedef struct st_LWAttrDouble4 {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy) ( LWAttrDouble4ID );
    LWAttributeID   (*baseAttr)( LWAttrDouble4ID );
    LWNameID        (*nameID)  ( LWAttrDouble4ID );
    void            (*setFlags)( LWAttrDouble4ID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)( LWAttrDouble4ID );
    const double *  (*value)   ( LWAttrDouble4ID );
    int             (*setValue)( LWAttrDouble4ID, const double value[ 4 ] );
} LWAttrDouble4;

/*!< Contains the wrapper for vparam. */

typedef struct st_LWAttrVParam {
    const void       *priv; /*!< Pointer to base record.   */
    void             *user; /*!< User defined data pointer.*/
    int             (*destroy)          ( LWAttrVParamID );
    LWAttributeID   (*baseAttr)         ( LWAttrVParamID );
    LWNameID        (*nameID)           ( LWAttrVParamID );
    void            (*setFlags)         ( LWAttrVParamID, unsigned int flags, unsigned int mask );
    unsigned int    (*getFlags)         ( LWAttrVParamID );
    double          (*value)            ( LWAttrVParamID );
    int             (*setValue)         ( LWAttrVParamID, const double );
    float           (*evaluate)         ( LWAttrVParamID, LWTime time );
    LWEnvelopeID    (*createEnvelope)   ( LWAttrVParamID, LWCStringASCII type );
    void            (*destroyEnvelope ) ( LWAttrVParamID );
    LWEnvelopeID    (*envelope)         ( LWAttrVParamID );
    void            (*setChannelGroup)  ( LWAttrVParamID, LWCStringASCII namedMemory, LWChanGroupID group );
    int             (*setEnvelopeState) ( LWAttrVParamID, int state );
    int             (*getEnvelopeState) ( LWAttrVParamID );
    LWCStringASCII  (*envelopeType)     ( LWAttrVParamID );
    LWTextureID     (*createTexture)    ( LWAttrVParamID );
    void            (*destroyTexture)   ( LWAttrVParamID );
    LWTextureID     (*texture)          ( LWAttrVParamID );
    void            (*setTextureContext)( LWAttrVParamID, LWCStringASCII namedMemory, LWTxtrContextID context, void *userdata );
    int             (*setTextureState)  ( LWAttrVParamID, int state );
    int             (*getTextureState)  ( LWAttrVParamID );
} LWAttrVParam;

/*!< Contains the wrapper for vparam3. */

typedef struct st_LWAttrVParam3 {
    const void           *priv; /*!< Pointer to base record.   */
    void                 *user; /*!< User defined data pointer.*/
    int                 (*destroy)            ( LWAttrVParam3ID );
    LWAttributeID       (*baseAttr)           ( LWAttrVParam3ID );
    LWNameID            (*nameID)             ( LWAttrVParam3ID );
    void                (*setFlags)           ( LWAttrVParam3ID, unsigned int flags, unsigned int mask );
    unsigned int        (*getFlags)           ( LWAttrVParam3ID );
    const double *      (*value)              ( LWAttrVParam3ID );
    int                 (*setValue)           ( LWAttrVParam3ID, const double value[ 3 ] );
    const float *       (*evaluate)           ( LWAttrVParam3ID, LWTime time );
    const LWEnvelopeID *(*createEnvelopes)    ( LWAttrVParam3ID, LWCStringASCII type );
    void                (*destroyEnvelopes)   ( LWAttrVParam3ID );
    const LWEnvelopeID *(*envelopes)          ( LWAttrVParam3ID );
    void                (*setChannelGroup)    ( LWAttrVParam3ID, LWCStringASCII namedMemory, LWChanGroupID group );
    int                 (*setEnvelopeStates)  ( LWAttrVParam3ID, int state[ 3 ] );
    int *               (*getEnvelopeStates)  ( LWAttrVParam3ID );
    LWCStringASCII      (*envelopesType)      ( LWAttrVParam3ID );
    LWTextureID         (*createTexture)      ( LWAttrVParam3ID );
    void                (*destroyTexture)     ( LWAttrVParam3ID );
    LWTextureID         (*texture)            ( LWAttrVParam3ID );
    void                (*setTextureContext)  ( LWAttrVParam3ID, LWCStringASCII namedMemory, LWTxtrContextID context, void *userdata );
    int                 (*setTextureState)    ( LWAttrVParam3ID, int state );
    int                 (*getTextureState)    ( LWAttrVParam3ID );
} LWAttrVParam3;

#define LWATTR_8CC_TEXT      LWLongID_('A','T','E','X','T','S','T','R') /*!< text      */
#define LWATTR_8CC_INT       LWLongID_('A','I','N','T','E','G','E','R') /*!< int       */
#define LWATTR_8CC_GRADIENT  LWLongID_('A','G','R','A','D','I','E','N') /*!< gradient  */
#define LWATTR_8CC_REFERENCE LWLongID_('A','R','E','F','E','R','E','N') /*!< reference */
#define LWATTR_8CC_IMAGE     LWLongID_('A','I','M','A','G','E','I','D') /*!< image id  */
#define LWATTR_8CC_FILE      LWLongID_('A','F','I','L','E','N','A','M') /*!< file      */
#define LWATTR_8CC_FLOAT     LWLongID_('A','F','L','O','A','T',' ',' ') /*!< float     */
#define LWATTR_8CC_FLOAT3    LWLongID_('A','F','L','O','A','T','3',' ') /*!< float[3]  */
#define LWATTR_8CC_FLOAT4    LWLongID_('A','F','L','O','A','T','4',' ') /*!< float[4]  */
#define LWATTR_8CC_DOUBLE    LWLongID_('A','D','O','U','B','L','E',' ') /*!< double    */
#define LWATTR_8CC_DOUBLE3   LWLongID_('A','D','O','U','B','L','E','3') /*!< double[3] */
#define LWATTR_8CC_DOUBLE4   LWLongID_('A','D','O','U','B','L','E','4') /*!< double[4] */
#define LWATTR_8CC_VPARAM    LWLongID_('A','V','P','A','R','A','M',' ') /*!< vparam    */
#define LWATTR_8CC_VPARAM3   LWLongID_('A','V','P','A','R','A','M','3') /*!< vparam3   */

/*!< This is the attribute utility function. */

typedef struct st_LWAttrUtility {
    LWAttributeID  (*create)       ( size_t maxEnum );
    int            (*destroy)      ( LWAttributeID );
    LWError        (*copy)         ( LWAttributeID dst, LWAttributeID src );
    int            (*load)         ( LWAttributeID, const LWLoadState * );
    int            (*save)         ( LWAttributeID, const LWSaveState * );
    int            (*setEvent)     ( LWAttributeID, LWAttrEventFunc, void *userData );
    void *         (*namedMemory)  ( LWAttributeID, LWCStringASCII name, size_t size );
    LWNameID       (*nextName)     ( LWAttributeID, LWGroupID, LWNameID );
    LWCStringASCII (*getNameFromId)( LWAttributeID, LWNameID );
    LWLongID       (*type)         ( LWAttributeID, LWNameID );
} LWAttrUtility;

/*!< These functions are used to create new attributes in attribute container. */

typedef struct st_LWAttrDefine {
    LWAttrStringID    (*createString)   ( LWAttributeID, LWCStringUTF8 name, LWCStringUTF8   value      );
    LWAttrIntID       (*createInt)      ( LWAttributeID, LWCStringUTF8 name, const int       value      );
    LWAttrGradientID  (*createGradient) ( LWAttributeID, LWCStringUTF8 name, LWCStringUTF8 tag, const unsigned int numberChannels );
    LWAttrReferenceID (*createReference)( LWAttributeID, LWCStringUTF8 name, const void     *value      );
    LWAttrImageID     (*createImage)    ( LWAttributeID, LWCStringUTF8 name, const LWImageID value      );
    LWAttrFileID      (*createFile)     ( LWAttributeID, LWCStringUTF8 name, LWCStringUTF8   value      );
    LWAttrFloatID     (*createFloat)    ( LWAttributeID, LWCStringUTF8 name, const float     value      );
    LWAttrFloat3ID    (*createFloat3)   ( LWAttributeID, LWCStringUTF8 name, const float     value[ 3 ] );
    LWAttrFloat4ID    (*createFloat4)   ( LWAttributeID, LWCStringUTF8 name, const float     value[ 4 ] );
    LWAttrDoubleID    (*createDouble)   ( LWAttributeID, LWCStringUTF8 name, const double    value      );
    LWAttrDouble3ID   (*createDouble3)  ( LWAttributeID, LWCStringUTF8 name, const double    value[ 3 ] );
    LWAttrDouble4ID   (*createDouble4)  ( LWAttributeID, LWCStringUTF8 name, const double    value[ 4 ] );
    LWAttrVParamID    (*createVParam)   ( LWAttributeID, LWCStringUTF8 name, const double    value      );
    LWAttrVParam3ID   (*createVParam3)  ( LWAttributeID, LWCStringUTF8 name, const double    value[ 3 ] );
} LWAttrDefine;

/*!< These functions are used to attach to an attribute in attribute container. */

typedef struct st_LWAttrAttach {
    LWAttrStringID    (*attachString)   ( LWAttributeID, LWCStringUTF8 name );
    LWAttrIntID       (*attachInt)      ( LWAttributeID, LWCStringUTF8 name );
    LWAttrGradientID  (*attachGradient) ( LWAttributeID, LWCStringUTF8 name );
    LWAttrReferenceID (*attachReference)( LWAttributeID, LWCStringUTF8 name );
    LWAttrImageID     (*attachImage)    ( LWAttributeID, LWCStringUTF8 name );
    LWAttrFileID      (*attachFile)     ( LWAttributeID, LWCStringUTF8 name );
    LWAttrFloatID     (*attachFloat)    ( LWAttributeID, LWCStringUTF8 name );
    LWAttrFloat3ID    (*attachFloat3)   ( LWAttributeID, LWCStringUTF8 name );
    LWAttrFloat4ID    (*attachFloat4)   ( LWAttributeID, LWCStringUTF8 name );
    LWAttrDoubleID    (*attachDouble)   ( LWAttributeID, LWCStringUTF8 name );
    LWAttrDouble3ID   (*attachDouble3)  ( LWAttributeID, LWCStringUTF8 name );
    LWAttrDouble4ID   (*attachDouble4)  ( LWAttributeID, LWCStringUTF8 name );
    LWAttrVParamID    (*attachVParam)   ( LWAttributeID, LWCStringUTF8 name );
    LWAttrVParam3ID   (*attachVParam3)  ( LWAttributeID, LWCStringUTF8 name );
} LWAttrAttach;

#endif