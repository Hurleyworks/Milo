/*
 * LWSDK Header File
 *
 * LWSIMULATION.H -- LightWave Simulation Support
 *
 *Copyright © 2024 LightWave Digital, Ltd. and its licensors. All rights reserved.
 *
 *This file contains confidential and proprietary information of LightWave Digital, Ltd.,
 *and is subject to the terms of the LightWave End User License Agreement (EULA).
 */

#ifndef LWSDK_SIMULATION_H
#define LWSDK_SIMULATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <lwtypes.h>
#include <lwxpanel.h>
#include <lwio.h>

#define LWSIMULATION_ENABLE_CALC    (1UL<<0UL)
#define LWSIMULATION_ENABLE_APPLY   (1UL<<1UL)

#define LWSIMULATION_REASON_PRIMARY     0
#define LWSIMULATION_REASON_SECONDARY   1

typedef struct st_LWSimulation
{
    int     version;        // Currently 1

    void*   (*startSim)     (void* simdata, LWTime start_time, LWTime end_time, unsigned int reason);
    int     (*getSimStep)   (void* simdata, void* passdata, LWTime* sim_time);
    int     (*startSimStep) (void* simdata, void* passdata, LWTime sim_time);
    int     (*endSimStep)   (void* simdata, void* passdata, LWTime sim_time);
    int     (*endSim)       (void* simdata, void* passdata);

    int     (*needReset)    (void* simdata);
    int     (*resetSim)     (void* simdata);

    LWCStringUTF8   (*descln)   (void* simdata);
    unsigned int    (*setflags) (void* simdata, unsigned int flags);
    LWXPanelID      (*options)  (void* simdata);
} LWSimulation;

typedef void* LWSimulationID;


#define LWSIMULATIONFUNCS_GLOBAL "LW Simulation Funcs"

typedef struct st_LWSimulationFuncs
{
    LWSimulationID  (*create)   (LWSimulation* sim, void* simdata);
    void            (*destroy)  (LWSimulationID simID);

    LWError         (*copy)     (LWSimulationID to_simID, LWSimulationID from_simID);
    LWError         (*load)     (LWSimulationID simID, const LWLoadState* ls);
    LWError         (*save)     (LWSimulationID simID, const LWSaveState* ss);

    unsigned int    (*flags)    (LWSimulationID simID, unsigned int set, unsigned int clear);

    void            (*trigger)  (void);
} LWSimulationFuncs;


#ifdef __cplusplus
}
#endif


#endif /* LWSDK_SIMULATION_H */