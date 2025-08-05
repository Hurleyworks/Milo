#pragma once

using sabi::PhysicsEngineState;
using sabi::SpaceTime;

struct updatePhysics
{
    updatePhysics (PhysicsEngineState state, uint32_t frameNumber = 0) :
        state (state),
        frame (frameNumber)
    {
    }

    QmsID id = QmsID::UpdatePhysics;
    QmsID realID = QmsID::UpdatePhysics;

    PhysicsEngineState state = PhysicsEngineState::Pause;
    uint32_t frame = 0;
};

struct updatePose
{
    updatePose (RenderableWeakRef weakNode) :
        weakNode (weakNode)
    {
        // start with the current SpaceTime
        if (!weakNode.expired())
            st = weakNode.lock()->getSpaceTime();
    }

    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::UpdatePose;

    RenderableWeakRef weakNode;

    // we need to keep a separate SpaceTime
    // for the command system
    SpaceTime st;
};

struct updatePhysicsProperties
{
    updatePhysicsProperties (RenderableWeakRef weakNode) :
        weakNode (weakNode)
    {
    
    }

    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::UpdatePhysicsProperties;

    RenderableWeakRef weakNode;
};

struct createOrUpdatePhantom
{
    createOrUpdatePhantom (RenderableWeakRef weakNode) :
        weakNode (weakNode)
    {
    }

    QmsID id = QmsID::CreateOrUpdatePhantom;
    QmsID realID = QmsID::CreateOrUpdatePhantom;

    RenderableWeakRef weakNode;
};