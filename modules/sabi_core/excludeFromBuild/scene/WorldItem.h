
#pragma once

#include "../../sabi_core.h"

using sabi::Renderable;
using sabi::RenderableData;
using sabi::RenderableNode;
using sabi::RenderableWeakRef;

class WorldItem : public Renderable
{
 public:
    static RenderableNode create() { return std::make_shared<WorldItem>(); }

 public:
    WorldItem() = default;
    ~WorldItem() = default;

    // geometry instances
    RenderableNode createInstance() override;
    bool isInstance() const override { return instancedFrom.lock() != nullptr; }
    void setInstancedFrom (RenderableNode node) override { instancedFrom = node; }
    RenderableNode getInstancedFrom() override { return instancedFrom.expired() ? nullptr : instancedFrom.lock(); }
    size_t getNumberOfInstances() const override { return instanceCount; }

    // physics phantom for collision free painting
    RenderableNode createPhysicsPhantom() override;
    void setPhantomFrom (RenderableNode node) override { phantomFrom = node; }
    RenderableNode getPhantomFrom() override { return phantomFrom.expired() ? nullptr : phantomFrom.lock(); }
    bool isPhantom() const override { return phantomFrom.lock() != nullptr; }

    RenderableNode getRestingOn() override { return restingOn.expired() ? nullptr : restingOn.lock(); }
    void setRestingOn (RenderableNode node) override { restingOn = node; }
   
    const RenderableData getData() const override
    {
        RenderableData d;
        d.clientID = clientID;
        d.sourceID = instancedFrom.expired() ? INVALID_ID : instancedFrom.lock()->getID();
        d.desc = desc;
        d.cgModel = cgModel;
        d.name = name;
        d.sceneID = id();
        d.spacetime = spacetime;
        d.state = state;
        const float* m = spacetime.worldTransform.matrix().data();
        for (int i = 0; i < 16; i++)
        {
            d.pose[i] = m[i];
        }
        return d;
    }

 private:
    RenderableWeakRef instancedFrom;
    size_t instanceCount = 0;

    RenderableWeakRef phantomFrom;

    RenderableWeakRef restingOn;
}; // end class WorldItem
