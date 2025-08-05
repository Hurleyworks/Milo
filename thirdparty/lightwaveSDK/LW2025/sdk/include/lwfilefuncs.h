/*
 * LWSDK Header File
 * Copyright 2012, NewTek, Inc.
 *
 * lwfilefuncs.H -- LightWave file functions.
 *
 * This header contains the basic declarations need to define the
 * LightWave file functions.
 */
#ifndef LWSDK_FILEFUNCS_H
#define LWSDK_FILEFUNCS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwtypes.h>

#define LWFILEFUNCS_GLOBAL "LWFileFunctions 2"

typedef enum en_LWFileType {
   LWFT_NONE,
   LWFT_NORMAL,
   LWFT_DIRECTORY,
   LWFT_UNKNOWN
} LWFileType;

typedef enum lwen_GSystemPath {
    LWGSP_CURRENT,
    LWGSP_CWD,
    LWGSP_TEMP,             /* r/w                                                         */
    LWGSP_PROGRAM,          /* r location of application files (bin folder)                */
    LWGSP_SYSTEM,
    LWGSP_EXENAME,          /* r                                                           */
    LWGSP_PLUGINDB,         /* r/w location of plugin database                             */
    LWGSP_INSTALL,          /* r root location (contains support folder and bin folder)    */
    LWGSP_INSTALL_SUPPORT,  /* r location of installation support folder                   */
    LWGSP_MACHINE_SUPPORT,  /* r machine scope support root location                       */
    LWGSP_USER_SUPPORT,     /* r/w user scope support root location (previously GSP_PREFS) */

    LWGSP_INSTALL_LICENSE,  /* r                            */
    LWGSP_MACHINE_LICENSE,  /* r                            */
    LWGSP_USER_LICENSE,     /* r/w (previously GSP_LICENSE) */

    LWGSP_INSTALL_LOCALE,   /* r (previously GSP_MSGTABLES) */
    LWGSP_MACHINE_LOCALE,   /* r                            */
    LWGSP_USER_LOCALE,      /* r/w                          */

    LWGSP_INSTALL_PLUGINS,  /* r (previously GSP_PLUGINS)   */
    LWGSP_MACHINE_PLUGINS,  /* r                            */
    LWGSP_USER_PLUGINS,     /* r/w                          */
    LWGSP_DESKTOP_DIRECTORY
} LWGSystemPath;


typedef struct st_FSysItem* LWFSysItemID;

/**
Sort order of a directory item's contents is handled through the navigation
routines below. Each takes one of the following options (sortopt) to indicate
the desired ordering. If 'asc' is true, sorting will will be in
ascending order (oldest date first, starting with 'A', etc.).
By default, sort order is undefined but may resemble name order.
*/

typedef enum en_LWFSYS_SORT
{
    LWFSYS_SORT_NONE,
    LWFSYS_SORT_NAME,
    LWFSYS_SORT_SIZE,
    LWFSYS_SORT_DATE,
    LWFSYS_SORT_TYPE
} LWFSysSort;

typedef enum en_LWFSYS_TYPE
{
    LWFSYS_TYPE_DIR,
    LWFSYS_TYPE_FILE,
    LWFSYS_TYPE_LINK,
    LWFSYS_TYPE_UNKNOWN
} LWFSysType;

/**
The following is used to obtain the name of an item. The 'nf' argument
takes one of the name format types to specify how the name should be
returned.
*/

typedef enum en_LWFSYS_NAMEFMT
{
    LWFSYS_NF_BASE,
    LWFSYS_NF_FULL,
    LWFSYS_NF_SHORTDISP
} LWFSysNameFmt;


/* All paths are utf8 encoded 0-terminated strings */
typedef struct st_LWFileFuncs {
    void        (*fileParseName  )(LWCStringUTF8 filename, LWMutableCStringUTF8 basename, LWMutableCStringUTF8 path);
    int         (*fileCreateDir  )(LWCStringUTF8 dirpath);
    int         (*fileDeleteFile )(LWCStringUTF8 filename);
    LWFileType  (*fileTestType   )(LWCStringUTF8 filename);
    int         (*fileLaunchExec )(LWCStringUTF8 filename, LWCStringUTF8 commandLine, unsigned int flags);
    LWCStringUTF8 (*fileSystemPath )(LWGSystemPath path_id);

    /* Removes the prefix portion of in_path if it starts with the prefix.
     * The out_path will be relative to prefix or it will be unchanged.
     * Example: in_path:"relpath/to/file" prefix:"c:/NewTek/Content/" out_path:"relpath/to/file"
     * Example: in_path:"c:/NewTek/Content/Scenes/file" prefix:"c:/NewTek/Content/" out_path:"Scenes/file"
     * The trailing separator in the prefix is optional.
     * Multiple path entries are allowed.
     */
    void        (*pathToRelative)(LWCStringUTF8 in_path, LWCStringUTF8 prefix, LWMutableCStringUTF8 out_path, unsigned int out_path_size);

    /* Assuming the in_path is a 'native' relative path, the given 'native' prefix is preprended to it.
     * Example: in_path:"relpath/to/file" prefix:"D:/Users/memyselfandi/Documents/Content" out_path:"D:/Users/memyselfandi/Documents/Content/relpath/to/file"
     * Example: in_path:"c:/NewTek/Content/Scenes/file" prefix:"d:/NewTek/Content/Objects/" out_path:"c:/NewTek/Content/Scenes/file"
     * The trailing separator in the prefix is optional.
     * Multiple path entries are allowed.
     */
    void        (*pathToAbsolute)(LWCStringUTF8 in_path, LWCStringUTF8 prefix, LWMutableCStringUTF8 out_path, unsigned int out_path_size);

    /* Convert file/path specifications between platform native and LW neutral.
     * Neutral paths use forward slash path separators with an option device prefix name followed by a colon.
     * Example:
     *          Neutral: absolute: "device:/path/to/file" relative: "/path/to/file" "file"
     *       OSX Native: absolute: "/Volumes/device/path/to/file" relative: "/path/to/file" "file"
     *   Windows Native: absolute: "device:\path\to\file" relative: "\path\to\file" "file" (device is a drive letter only)
     *     Linux Native: absolute: "/mnt/device/path/to/file" relative: "/path/to/file" "file"
     * Multiple path entries are allowed.
     */
    void        (*pathToNeutral)(LWCStringUTF8 in_path, LWMutableCStringUTF8 out_path, unsigned int out_path_size); /* in_path assumed native */
    void        (*pathToNative)(LWCStringUTF8 in_path, LWMutableCStringUTF8 out_path, unsigned int out_path_size); /* in_path assumed neutral */

    /* determine if the given path is compatible with LW neutral format. If not, it may be native or malformed.
     * Multiple path entries are allowed.
     */
    int         (*pathIsNeutral)(LWCStringUTF8 in_path);

    /* determine if the given path has multiple path entries (semi-colon separators) */
    /* e.g. "device:/path/to/file1;file2;relpath/to/file3" */
    int         (*pathHasMultiple)(LWCStringUTF8 in_path);

    /* construct a multi-entry string from an array of single paths */
    void        (*pathJoinMultiple)(LWCStringUTF8 in_paths[], unsigned int in_num_paths, LWMutableCStringUTF8 out_path, unsigned int out_path_size);

    /* deconstruct a multi-path string into an array of single paths, which must be freed with pathFreeArray() */
    void        (*pathSeparateMultiple)(LWCStringUTF8 in_path, LWMutableCStringUTF8 *out_paths, unsigned int *out_num_paths);

    /* free an array of paths constructed with pathSeparateMultiple */
    void        (*pathFreeArray)(LWMutableCStringUTF8 in_paths[], unsigned int in_num_paths);

    /*
     * Implements standard fallback behavior for convenience, which is:
     * 1. If the path is relative to the given dirtype's directory, make out_path relative to that
     * 2. If the path is relative to the content directory, make out_path relative to that
     * 3. If the path is neither relative to the dirtype directory nor the content directory, copy in_path to out_path
     *
     * @param dirtype One of the "LWFTYPE_" defines from lwhost.h
     */
    void        (*pathToRelativeStdFallbacks)(LWCStringUTF8 dirtype, LWCStringUTF8 in_path, LWMutableCStringUTF8 out_path, unsigned int out_path_size);

    /*
    * Implements standard fallback behavior for convenience, which is:
    * 1. If the path made absolute given the dirtype's directory, and the file exists, return it in out_path
    * 2. Make out_path absolute to the content directory
    *
    * @param dirtype One of the "LWFTYPE_" defines from lwhost.h
    */
    void        (*pathToAbsoluteStdFallbacks)(LWCStringUTF8 dirtype, LWCStringUTF8 in_path, LWMutableCStringUTF8 out_path, unsigned int out_path_size);

    /* Hierarchical file system scanning */
    LWFSysItemID    (*fSysCreate)   ();
    void            (*fSysDestroy)  (LWFSysItemID);
    LWFSysItemID    (*fSysFindPath) (LWFSysItemID, LWCStringUTF8 path);
    int             (*fSysCount)    (LWFSysItemID, int type, int ignore_filter);
    LWFSysItemID    (*fSysSubFile)  (LWFSysItemID, int index, int sort_opt, int ascend, int ignore_filter);
    LWCStringUTF8   (*fSysName)     (LWFSysItemID, int name_format);

    int         (*filePathCompose)(LWMutableCStringUTF8 filename, LWCStringUTF8 basename, LWCStringUTF8 path);
    int         (*fileIsAbsolute )(LWCStringUTF8 filename);
    LWMutableCStringUTF8 (*fileLocate)(LWMutableCStringUTF8 local, LWCStringUTF8 neutral, LWCStringUTF8 type );

} LWFileFuncs;

#ifdef __cplusplus
}
#endif

#endif