#pragma once

// LWO3Chunk: Represents a chunk in the LWO3 file structure
//
// Chunks are leaf nodes in the LWO3 file structure tree. They contain actual data.
// This class is part of the Composite Pattern, representing the "Leaf" role.

//#include "LWO3Element.h"
//#include "LWO3Visitor.h"

class LWO3Chunk : public LWO3Element
{
 public:
    LWO3Chunk (uint32_t id, size_t offset = 0) :
        LWO3Element (id, offset) {}

    void accept (LWO3Visitor& visitor) const override
    {
        visitor.visitChunk (*this);
    }

    bool isForm() const override { return false; }

    const std::vector<uint8_t>& getData() const { return data_; }
    void setData (const std::vector<uint8_t>& data) { data_ = data; }

 private:
    std::vector<uint8_t> data_;
};