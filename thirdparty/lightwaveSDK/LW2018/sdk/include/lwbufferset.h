/**
 *      @brief BufferNames
 *
 *      @short This is the header for the buffer names module.
 */

#ifndef LAYOUT_BUFFERNAMES_H
#define LAYOUT_BUFFERNAMES_H

#include <lwtypes.h>
#include <lwserver.h>
#include <lwio.h>

/* Make sure we have true false booleans. */

#if( !defined( FALSE ) )
    #define FALSE 0
#endif
#if( !defined( TRUE ) )
    #define TRUE !FALSE
#endif

/* Buffer catagory names. */

#define LWStandardBufferGroup               "Standard"
#define LWExtraBufferGroup                  "Extra"
#define LWCustomBufferGroup                 "Custom"
#define LWLightsBufferGroup                 "Lights"

/* Buffer Render/Save flags. */

#define LWBUFFERSETFLAG_IGNORE         0x00
#define LWBUFFERSETFLAG_RENDER         0x01
#define LWBUFFERSETFLAG_SAVE           0x02
#define LWBUFFERSETFLAG_USECUSTOMNAME  0x04
#define LWBUFFERSETFLAG_USEPREFOUTPUT  0x08
#define LWBUFFERSETFLAG_NOISEFILTER    0x10
#define LWBUFFERSETFLAG_ADAPTIVESAMPLE 0X20
#define LWBUFFERSETFLAG_INVERT_Z       0x40

typedef struct LWBufferList* LWBufferListID;
typedef struct LWBufferSet* LWBufferSetID;

/*
Note: Unofficial access to Layout's internal Buffer Set can be had by
      passing -1 to the create call. This returned Buffer Set should
      NOT be destroyed, as Layout will manage it.
*/
#define BufferSet_INTERNAL -1

enum BufferSetType {
    BufferSet_ALL,
    BufferSet_LIGHT,
    BufferSet_MAX
};

enum ReconstructionFilterType {
    ReconstructionFilter_BOX,
    ReconstructionFilter_CIRCLE,
    ReconstructionFilter_TRIANGLE,
    ReconstructionFilter_GAUSSIAN,
    ReconstructionFilter_MITCHELL,
    ReconstructionFilter_LANCZOSSINC,
    ReconstructionFilter_MAX
};

#define LWBUFFERLIST_GLOBAL "Buffer List Global"

typedef int LWBufferEventFunc( void *userData, void *bufferSet, unsigned int event, void *eventData );

#define LWBEVENT_CHANGED 1
#define LWBEVENT_SYNCED  2

typedef struct st_LWBufferSetGlobal {
    LWBufferSetID  (*create          )( int set_type );   /*!< Creates a buffer set. */
    int            (*destroy         )( LWBufferSetID set );
    int            (*copy            )( LWBufferSetID dest, LWBufferSetID sorc, int addRecords );
    int            (*load            )( LWBufferSetID set, const LWLoadState *load );
    int            (*save            )( LWBufferSetID set, const LWSaveState *save );
    int            (*clear           )( LWBufferSetID set );
    int            (*rebase          )( LWBufferSetID set );

    int            (*numberOfGroups  )( LWBufferSetID set );
    int            (*numberOfItems   )( LWBufferSetID set, int group );
    LWBufferListID (*bufferRecord    )( LWBufferSetID set, int group, int item );
    const char *   (*groupName       )( LWBufferSetID set, int group );
    const char *   (*getBufferName   )( LWBufferListID record );
    int            (*setBufferFlags  )( LWBufferListID record, int flags );
    int            (*getBufferFlags  )( LWBufferListID record );
    int            (*setCustomName   )( LWBufferListID record, const char *name );
    const char *   (*getCustomName   )( LWBufferListID record );
    int            (*setSaveFormat   )( LWBufferListID record, const char *format_name );
    const char *   (*getSaveFormat   )( LWBufferListID record );
    int            (*setColorSpace   )( LWBufferListID record, const char *color_space );
    const char *   (*getColorSpace   )( LWBufferListID record );
    int            (*setOutputPath   )( LWBufferListID record, const char *path );
    const char *   (*getOutputPath   )( LWBufferListID record );
    int            (*setFilename     )( LWBufferListID record, const char *filename );
    const char *   (*getFilename     )( LWBufferListID record );
    int            (*setReconstructionFilter)(LWBufferListID record, int filter);
    int            (*getReconstructionFilter)(LWBufferListID record);

    LWBufferListID (*addLightGroup   )( LWBufferSetID set, const char *name );
    int            (*renameLightGroup)( LWBufferSetID set, LWBufferListID record, const char* name );
    int            (*removeLightGroup)( LWBufferSetID set, LWBufferListID record );
    LWBufferSetID  (*lightBufferSet  )( LWBufferListID record );
    LWBufferSetID  (*bufferSetForLight )( LWBufferSetID set, LWItemID light_id );

    int            (*setBufferEvent )( LWBufferSetID set, LWBufferEventFunc *userFunc, void *userData );
    int            (*sendBufferEvent)( LWBufferSetID set, int eventCode );
} LWBufferSetGlobal;

#endif