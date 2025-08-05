#pragma once

// LWO3Visitor: Abstract base class for visitors of the LWO3 structure
//
// This class defines the interface for concrete visitors that can traverse
// and operate on the LWO3 file structure. It's part of the Visitor Pattern implementation.

//#include <wabi_core/wabi_core.h>

class LWO3Form;
class LWO3Chunk;

class LWO3Visitor
{
 public:
    virtual ~LWO3Visitor() = default;
    virtual void visitForm (const LWO3Form& form) = 0;
    virtual void visitChunk (const LWO3Chunk& chunk) = 0;
};