/**
 *      @brief lwpreferencesystem.h
 *
 *      @short Contains the preferences global.
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 *
 */

#ifndef LWSDK_PREFERENCESYSTEM_H
#define LWSDK_PREFERENCESYSTEM_H

#include <lwtypes.h>
#include <lwio.h>


#define LWPREFERENCESYSTEM_GLOBAL "PreferenceSystem"

typedef enum {
    LWPrefTypeNone = 0,
    LWPrefTypeInt,
    LWPrefTypeFloat,
    LWPrefTypeString,
    LWPrefTypeFloat3,
    LWPrefTypeProxy
} LWPreferenceSystemDataType;

typedef enum {
    LWPrefEventPublished = 1,
    LWPrefEventValueChanged,
    LWPrefEventValueLoaded,
    LWPrefEventValueReset,
    LWPrefEventExtraDataChanged,
    LWPrefEventHintsChanged,
    LWPrefEventPreRemove,
    LWPrefEventPostRemove,
    LWPrefEventCallbacksSet
} LWPreferenceSystemEvent;

#define LWPrefPanelFlagSolo         0x01
#define LWPrefPanelFlagSoloKeepSize 0x02
#define LWPrefPanelFlagSoloWithTree 0x04

#define LWPrefPopupFlagHighlight    0x0200
#define LWPrefPopupFlagDisabled     0x0100
#define LWPrefPopupFlagChecked      0xC000

#define LWPrefSetResult_Validated   -2
#define LWPrefSetResult_NotSet      -1
#define LWPrefSetResult_NotFound    0
#define LWPrefSetResult_Success     1
#define LWPrefSetResult_NoChange    2

#define LWPrefHostBranch  "$/"
#define LWPrefSceneBranch "/Current Scene/"

typedef struct st_LWPreferenceSystemFloat3 {
    double first;  /* X-H-R */
    double second; /* Y-P-G */
    double third;  /* Z-B-B */
} LWPreferenceSystemFloat3;

typedef struct st_LWPreferenceSystemParameterInfo* LWPreferenceSystemParameterInfoPtr;
typedef struct st_LWPreferenceSystemCallbacks {
    void *userData;
    void(*buttonClick)(const LWPreferenceSystemParameterInfoPtr info);
    int(*itemCount)(const LWPreferenceSystemParameterInfoPtr info);
    LWCStringUTF8(*itemName)(const LWPreferenceSystemParameterInfoPtr info, int item);
    int(*itemState)(const LWPreferenceSystemParameterInfoPtr info, int item);
} LWPreferenceSystemCallbacks;


typedef struct st_LWPreferenceSystemParameterInfo {
    LWCStringASCII key;
    LWPreferenceSystemDataType type;
    unsigned int ident;
    LWCStringUTF8 l10nContext;
    LWCStringUTF8 hints;
    float order;
    LWPreferenceSystemCallbacks *callbacks;
} LWPreferenceSystemParameterInfo;


typedef int LWPreferanceSystemValidatorFunc(void *userData, LWCStringASCII parameterKey, const LWPreferenceSystemParameterInfo *info, LWPreferenceSystemEvent event, void *eventData);
typedef void LWPreferenceSystemEventFunc(void *userData, LWCStringASCII parameterKey, const LWPreferenceSystemParameterInfo *info, LWPreferenceSystemEvent event, void *eventData);


typedef struct st_LWPreferenceSystem {
    int (*parameterExists      )(LWCStringASCII path);
    unsigned int (*publishParameter     )(LWCStringASCII path, LWCStringASCII parameterName, LWPreferenceSystemDataType type, unsigned int ident, LWCStringUTF8 l10nContext, LWCStringUTF8 hints);

    int (*setValidator         )(LWCStringASCII path, LWCStringASCII parameterName, LWPreferanceSystemValidatorFunc *validateFunc, void *userData);
    int (*setCallbacks         )(LWCStringASCII path, LWCStringASCII parameterName, const LWPreferenceSystemCallbacks *callbacks);
    int (*setEventHandler      )(LWCStringASCII path, LWPreferenceSystemEventFunc *userFunc, void *userData);

    int (*removeParameters     )(LWCStringASCII path);

	unsigned int (*getParameterCount)(LWCStringASCII prefix);
    LWCStringASCII (*getParameterName)(LWCStringASCII prefix, unsigned int index);

	LWError (*savePublications)(LWCStringASCII path, LWCStringUTF8 pubFilePath);
    LWError (*loadPublications)(LWCStringUTF8 pubFilePath);
    LWError (*saveValues)(LWCStringASCII path, LWCStringUTF8 valueFile);
    LWError (*loadValues)(LWCStringUTF8 valueFile);
    LWError (*saveValuesToBlock)(LWCStringASCII path, const LWSaveState *save);
    LWError (*loadValuesFromBlock)(const LWLoadState *load);

    const LWPreferenceSystemParameterInfo* (*getParameterInfo)(LWCStringASCII path, LWCStringASCII parameterName);

    int (*getIntValue)(LWCStringASCII path, LWCStringASCII parameterName);
    double (*getFloatValue)(LWCStringASCII path, LWCStringASCII parameterName);
    LWCStringUTF8 (*getStringValue)(LWCStringASCII path, LWCStringASCII parameterName);
    const LWPreferenceSystemFloat3* (*getFloat3Value)(LWCStringASCII path, LWCStringASCII parameterName);

    int (*setIntValue)(LWCStringASCII path, LWCStringASCII parameterName, int value);
    int (*setFloatValue)(LWCStringASCII path, LWCStringASCII parameterName, double value);
    int (*setStringValue)(LWCStringASCII path, LWCStringASCII parameterName, LWCStringUTF8 value);
    int (*setFloat3Value)(LWCStringASCII path, LWCStringASCII parameterName, const LWPreferenceSystemFloat3 *value);

    void (*resetValues)(LWCStringASCII path);

    LWCStringUTF8 (*getExtraData)(LWCStringASCII path, LWCStringASCII parameterName);
    int (*setExtraData)(LWCStringASCII path, LWCStringASCII parameterName, LWCStringUTF8 value, int notify);

    LWCStringUTF8 (*dataApi)(LWCStringUTF8 cmd, LWCStringUTF8 data);

    void (*openPanel)(LWCStringASCII path, LWCStringUTF8 search, int flags);
    // A NULL path will close or test the main Preferences panel
    // Providing the path will close or test a panel opened in "solo mode"
    void (*closePanel)(LWCStringASCII path);
    int (*isPanelOpen)(LWCStringASCII path);

} LWPreferenceSystem;

#endif