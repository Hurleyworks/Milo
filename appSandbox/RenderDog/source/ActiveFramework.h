#pragma once

#include <qms_core/qms_core.h>
//#include <world_core/world_core.h>
//#include <image_core/image_core.h>
#include <milo_core/milo_core.h>
//#include <physics_core/physics_core.h>
//#include <geometry_core/geometry_core.h>
#include "ActiveDreamer.h"

struct ActiveFramework
{
    ActiveFramework() = default;
    ~ActiveFramework()
    {
        render.done();
     //   imaging.done();
     //   geometry.done();
      //  world.done();
      //  newton.done();
        dreamer.done();
    }

    void init (const PropertyService& properties)
    {
        // initialize messengers
        messengers.render = render.getMessenger();
       // messengers.imaging = imaging.getMessenger();
       // messengers.geometry = geometry.getMessenger();
        //messengers.world = world.getMessenger();
        //messengers.physics = newton.getMessenger();
        messengers.dreamer = dreamer.getMessenger();

        // initialize active object modules
        messengers.render.send (QMS::init (messengers, properties));
       // messengers.imaging.send (QMS::init (messengers,properties));
       // messengers.geometry.send (QMS::init (messengers, properties));
        //messengers.world.send (QMS::init (messengers, properties));
        //messengers.physics.send (QMS::init (messengers, properties));
        messengers.dreamer.send (QMS::init (messengers, properties));
    }

    // active object modules
    ActiveRender render;
   // ActiveImage imaging;
   // ActiveGeometry geometry;
    //ActiveWorld world;
   // ActivePhysics newton;
    ActiveDreamer dreamer;

    MessageService messengers;

    void shutdown()
    {
        render.done();
    }
};
