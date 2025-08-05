#include "WorldItem.h"


RenderableNode WorldItem::createInstance()
{
    RenderableNode instance = WorldItem::create();

    instance->setDescription (Renderable::description());
    instance->setSpacetime (Renderable::getSpaceTime());
    instance->setState (Renderable::getState());

    // instances have no mesh!
    instance->setModel(nullptr);
    instance->setInstancedFrom (getPtr());

    ++instanceCount;
    instance->setName (getName() + "_instance_" + std::to_string (instanceCount));

    return instance;
}

RenderableNode WorldItem::createPhysicsPhantom()
{
    RenderableNode phantom = WorldItem::create();

    phantom->setDescription (Renderable::description());
    phantom->setSpacetime (Renderable::getSpaceTime());
    phantom->setState (Renderable::getState());
    // phantoms can use their phantom from geometry
    phantom->setModel (nullptr);
    phantom->setPhantomFrom (getPtr());
    phantom->setName (getName() + "_phantom");

    return phantom;
}
