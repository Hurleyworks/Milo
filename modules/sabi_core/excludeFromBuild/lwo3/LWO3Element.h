#pragma once

// Composite Pattern Explanation:
//
// The Composite Pattern is a structural design pattern that allows composing objects into tree
// structures to represent part-whole hierarchies. It enables clients to treat individual objects
// and compositions of objects uniformly.
//
// In the context of the LWO3 file structure:
//
// 1. LWO3Element is the component interface, declaring operations common to both simple and
//    complex elements of the composition.
//
// 2. LWO3Chunk is the leaf, representing end objects of the composition. A leaf has no children.
//
// 3. LWO3Form is the composite, representing complex elements that may have children. It stores
//    child components and implements child-related operations.
//
// This pattern is particularly useful for the LWO3 format because:
// - It reflects the natural structure of the LWO3 file format, which consists of nested forms
//   and chunks.
// - It allows for recursive processing of the file structure, as each form can contain other
//   forms or chunks.
// - It provides a uniform interface for working with both simple (chunks) and complex (forms)
//   elements, simplifying client code that processes the LWO3 structure.
//
// The Visitor Pattern is often used in conjunction with the Composite Pattern to perform
// operations on the composite structure, as seen in the `accept` methods and the LWO3Visitor class.

// LWO3Element: Abstract base class for LWO3 file structure elements
//
// This class represents the common interface for both LWO3 chunks and forms.
// It's part of the Composite Pattern implementation for LWO3 file structure.
//#include "LWO3Defs.h"

class LWO3Visitor;

class LWO3Element
{
 public:
    LWO3Element (uint32_t id, size_t offset = 0) :
        id_ (id),
        fileOffset_ (offset) {}
    virtual ~LWO3Element() = default;

    virtual void accept (LWO3Visitor& visitor) const = 0;
    virtual bool isForm() const = 0;

    uint32_t getId() const { return id_; }
    size_t getFileOffset() const { return fileOffset_; }

 protected:
    uint32_t id_;
    size_t fileOffset_;
};




