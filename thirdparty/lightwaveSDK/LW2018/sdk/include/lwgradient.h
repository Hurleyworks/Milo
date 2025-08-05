/*
 * LWSDK Header File
 *
 *  LWGRADIENT.H (LWSDK)
 *
 *Copyright © 2018 NewTek, Inc. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of NewTek, Inc.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */
#ifndef LWSDK_GRADIENT_H_included
#define LWSDK_GRADIENT_H_included

#ifdef __cplusplus
extern "C" {
#endif

#include <lwio.h>

typedef struct st_LWGradient *LWGradientRef;
typedef struct st_LWGradientKey *LWGradientKeyRef;
typedef struct st_LWGradientKeyIterator *LWGradientKeyIteratorRef;
typedef double LWGradientKeyDType;
typedef double LWGradientKeyIType;

/* key curves affect the incoming interpolation behavior */
#define LWGRADIENTKEY_CURVE_STEPPED 0
#define LWGRADIENTKEY_CURVE_LINEAR  1
#define LWGRADIENTKEY_CURVE_HERMITE 2
#define LWGRADIENTKEY_CURVE_BEZIER  3

/* curves behaviors affect the extrapolation when outside the range the existing keys */
#define LWGRADIENT_CURVEBEHAVIOR_NONE       0
#define LWGRADIENT_CURVEBEHAVIOR_HOLD       1
#define LWGRADIENT_CURVEBEHAVIOR_REPEAT     2
#define LWGRADIENT_CURVEBEHAVIOR_PINGPONG   3

/* key flags */
#define LWGRADIENTKEY_FLAG_INACTIVE_MASK (1<<0) /* inactive keys are not evaluated */
#define LWGRADIENTKEY_FLAG_LOCKED_MASK   (1<<1) /* locked keys can not be modified */
#define LWGRADIENTKEY_FLAG_FOCUS_MASK    (1<<2) /* focus keys are more interesting at the moment */

/* immutable (read-only) objects */
typedef const struct st_LWImmutableGradient *LWImmutableGradientRef;
typedef const struct st_LWImmutableGradientKey *LWImmutableGradientKeyRef;

// --- Version 1 ---
typedef struct st_LWGradientFuncs_V1
{
    /** create an empty gradient */
    LWGradientRef (*create)(void);

    /** destory an existing gradient */
    void (*destroy)(LWGradientRef gradient);

    /** copy the source to the target. The target becomes a duplicate of the source (no key merging) */
    void (*copy)(LWGradientRef gradient_target, LWGradientRef gradient_source);

    /** 1 or more dependent values per key */
    void (*setDValueCount)(LWGradientRef gradient, unsigned short count);
    void (*getDValueCount)(LWGradientRef gradient, unsigned short *count);

    /** set/get the amount of bytes allocated per key for client-managed data */
    void (*setClientSize)(LWGradientRef gradient, unsigned short size);
    void (*getClientSize)(LWGradientRef gradient, unsigned short *size);

    /** set/get the behavior of the curve for domain that occurs before the first key */
    void (*setPreBehavior)(LWGradientRef gradient, unsigned short behavior);
    void (*getPreBehavior)(LWGradientRef gradient, unsigned short *behavior);

    /** set/get the behavior of the curve for domain that occurs after the last key */
    void (*setPostBehavior)(LWGradientRef gradient, unsigned short behavior);
    void (*getPostBehavior)(LWGradientRef gradient, unsigned short *behavior);

    /** create a key */
    LWGradientKeyRef (*keyCreate)(LWGradientRef gradient);

    /** destroy a key that has already been removed from its gradient */
    void (*keyDestroy)(LWGradientKeyRef key);

    /** set a key's indepedent value
     * The key may be reinserted to maintain proper sort order in the gradient.
     * Quantization of the independent value must be handled outside this API.
     */
    void (*keySetIValue)(LWGradientKeyRef key, LWGradientKeyIType value);

    void (*keyGetIValue)(LWGradientKeyRef key, LWGradientKeyIType *value);

    /** set the key's flag bits (LWGRADIENTKEY_FLAG_...) only for those bits in the mask */
    void (*keySetFlags)(LWGradientKeyRef key, unsigned short mask_bits, unsigned short state_bits);

    void (*keyGetFlags)(LWGradientKeyRef key, unsigned short *state_bits);

    /** set the key's curve type (LWGRADIENTKEY_CURVE_...)
     * per-curve data not supported
     */
    void (*keySetCurve)(LWGradientKeyRef key, unsigned short curve);

    void (*keyGetCurve)(LWGradientKeyRef key, unsigned short *curve);

    /* query the key's owner. NULL if not part of any gradient. */
    LWGradientRef (*keyOwner)(LWGradientKeyRef key);

    /** set all dependent values via a provided array */
    void (*keySetDValues)(LWGradientKeyRef key, const LWGradientKeyDType *dvalue_array);

    /** get all dependent values via a provided array */
    void (*keyGetDValues)(LWGradientKeyRef key, LWGradientKeyDType *dvalue_array);

    /** set a single dependent value having the given index (0 to dvalue_count-1) */
    void (*keySetDValue)(LWGradientKeyRef key, LWGradientKeyDType dvalue, unsigned short index);

    /** get a single dependent value having the given index (0 to dvalue_count-1) */
    void (*keyGetDValue)(LWGradientKeyRef key, LWGradientKeyDType *dvalue, unsigned short index);

    /** access to a key's client data.
     * The number of bytes it represents must have been set via setClientSize()
     * If never set, this will return NULL.
     */
    void *(*keyAccessClientData)(LWGradientKeyRef key);

    /* create a sequential-access key iterator */
    LWGradientKeyIteratorRef (*keyIteratorCreate)(LWGradientRef gradient);

    /* NULL is returned when not available */
    LWGradientKeyRef (*keyIteratorFirst)(LWGradientKeyIteratorRef it);
    LWGradientKeyRef (*keyIteratorLast)(LWGradientKeyIteratorRef it);
    LWGradientKeyRef (*keyIteratorNext)(LWGradientKeyIteratorRef it);
    LWGradientKeyRef (*keyIteratorPrev)(LWGradientKeyIteratorRef it);

    /** destroy a key iterator */
    void (*keyIteratorDestroy)(LWGradientKeyIteratorRef iterator);

    /** add the unowned key to a gradient
     * A key is owned by the gradient it has been added to.
     * Do not insert a key that is already owned by the gradient
     */
    void (*insertKey)(LWGradientRef gradient, LWGradientKeyRef key);

    /** remove the key from the gradient.  It still exists but will be unowned */
    void (*removeKey)(LWGradientRef gradient, LWGradientKeyRef key);

    /* clear/remove all keys */
    void (*clearKeys)(LWGradientRef gradient);

    /* get the number of keys in the gradient. */
    unsigned int (*numKeys)(LWGradientRef gradient);

    /* get the key for a given index [0 to numKeys()-1] */
    LWGradientKeyRef (*keyAtIndex)(LWGradientRef gradient, unsigned int index);

    /** sort the gradient keys based on current indepedent values */
    void (*sort)(LWGradientRef gradient);

    /** evaluate the gradient for a given independent value.
     * keys must be sorted; otherwise, out-of-sequence keys will be skipped.
     */
    int (*evaluate)(LWGradientRef gradient, LWGradientKeyIType ivalue, LWGradientKeyDType *dvalues);

    /** save the gradient */
    /** The state saves the contents of the gradient, not the existence of it */
    LWError (*save)(const LWSaveState *state, const LWGradientRef gradient);

    /** load the gradient */
    /** The state is assumed to be at the start of the inside of the data block. */
    LWError (*load)(const LWLoadState *state, LWGradientRef gradient);

    /** Freezing a gradient creates an immutable (read-only) version that contains only
     * active keys for the purpose of evaluation
     * The immutable version can exist complete independent and can not be modified
     */
	LWImmutableGradientRef (*freeze)(LWGradientRef gradient);

    /* destroy a frozen gradient */
	void (*immutableDestroy)(LWImmutableGradientRef fgradient);

    /* return key count in immutable gradient */
	unsigned int (*immutableKeyCount)(LWImmutableGradientRef fgradient);

    /* return dependent value count of each immutable gradient key */
	unsigned short (*immutableDValueCount)(LWImmutableGradientRef fgradient);

    /* get pre behavior */
    unsigned short (*immutablePreBehavior)(LWImmutableGradientRef fgradient);

    /* get post behavior */
    unsigned short (*immutablePostBehavior)(LWImmutableGradientRef fgradient);

    /* return immutable key reference at a index (range 0 to n-1) */
	LWImmutableGradientKeyRef (*immutableKeyAtIndex)(LWImmutableGradientRef fgradient, unsigned int index);

    /* get key curve */
    unsigned short (*immutableKeyCurve)(LWImmutableGradientKeyRef fkey);

    /* return reference to independent value of an immutable key */
	const LWGradientKeyIType *(*immutableKeyIValue)(LWImmutableGradientKeyRef fkey);

    /* return reference to dependent value array of an immutable key */
	const LWGradientKeyDType *(*immutableKeyDValues)(LWImmutableGradientKeyRef fkey);

    /* evaluate the immutable gradient for a given independent value */
	void (*immutableEvaluate)(LWImmutableGradientRef fgradient, LWGradientKeyIType ivalue, LWGradientKeyDType *dvalues);

} LWGradientFuncs_V1;

#define LWGRADIENTFUNCS_GLOBAL_V1 "LWGradientFuncs 1"

/* LWGradient notifies clients of changes via the LWComRing API */
#define LWCOMRING_TOPIC_LWGRADIENT "LWGradient"     /* topic clients can listen to */

#define LWCOMRING_EVENT_LWGRADIENT_CURVE 		3 		/* curve changed [LWGradientRef] */
#define LWCOMRING_EVENT_LWGRADIENT_SELECTION 	4 		/* key selection changed [LWGradientRef] */
#define LWCOMRING_EVENT_LWGRADIENT_DESTROY   	5 		/* about to destroy [LWGradientRef] */
/* ---------------------- */

/* --- Latest Version --- */
typedef LWGradientFuncs_V1 LWGradientFuncs;
#define LWGRADIENTFUNCS_GLOBAL LWGRADIENTFUNCS_GLOBAL_V1
/* ---------------------- */

#ifdef __cplusplus
}
#endif

#endif
