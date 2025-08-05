#pragma once

// LWO3Form: Represents a container element in the LWO3 file structure
// Forms can contain other forms or chunks and track their file offset for debugging
// Serves as the "Composite" component in the Composite Pattern

class LWO3Form : public LWO3Element
{
 public:
    LWO3Form (uint32_t id, size_t offset = 0) :
        LWO3Element (id, offset) {}

    void accept (LWO3Visitor& visitor) const override
    {
        visitor.visitForm (*this);
        for (const auto& child : children_)
        {
            child->accept (visitor);
        }
    }

    bool isForm() const override { return true; }

    void addChild (std::unique_ptr<LWO3Element> child)
    {
        children_.push_back (std::move (child));
    }

    const std::vector<std::unique_ptr<LWO3Element>>& getChildren() const
    {
        return children_;
    }

 private:
    std::vector<std::unique_ptr<LWO3Element>> children_;
};