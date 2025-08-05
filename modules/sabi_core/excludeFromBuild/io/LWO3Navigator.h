#pragma once

// LWO3Navigator: Utility class for navigating and debugging LWO3 file structures
// Provides methods to find, validate and inspect forms/chunks using their file offsets


class LWO3Navigator
{
 public:
    // Search modes for findElementsById
    enum class SearchMode
    {
        ALL,          // Find all matching elements
        FIRST,        // Find first match only
        WITHIN_OFFSET // Find elements within offset range
    };

    struct SearchResult
    {
        const LWO3Element* element;
        std::vector<const LWO3Element*> path; // Path from root to element
    };

    // Find elements by ID with optional offset constraints
    std::vector<SearchResult> findElementsById (
        const LWO3Form* root,
        uint32_t targetId,
        SearchMode mode = SearchMode::ALL,
        size_t startOffset = 0,
        size_t endOffset = std::numeric_limits<size_t>::max())
    {
        std::vector<SearchResult> results;
        std::vector<const LWO3Element*> currentPath;
        findElementsByIdImpl (root, targetId, mode, startOffset, endOffset, currentPath, results);
        return results;
    }

    // Get element at specific file offset
    const LWO3Element* findElementAtOffset (const LWO3Form* root, size_t targetOffset)
    {
        if (!root || root->getFileOffset() > targetOffset)
        {
            return nullptr;
        }

        // Check each child
        for (const auto& child : root->getChildren())
        {
            if (child->getFileOffset() == targetOffset)
            {
                return child.get();
            }

            // If child is a form and target is within its range, recurse
            if (child->isForm())
            {
                if (auto* found = findElementAtOffset (
                        static_cast<const LWO3Form*> (child.get()),
                        targetOffset))
                {
                    return found;
                }
            }
        }
        return nullptr;
    }

    // Print hierarchy with offsets for debugging
    void dumpHierarchy (const LWO3Form* root, std::ostream& out = std::cout)
    {
        dumpHierarchyImpl (root, 0, out);
    }

    // Get parent element of a given offset
    const LWO3Form* findParentOfOffset (
        const LWO3Form* root,
        size_t childOffset,
        std::vector<const LWO3Form*>* path = nullptr)
    {
        if (!root || childOffset < root->getFileOffset())
        {
            return nullptr;
        }

        for (const auto& child : root->getChildren())
        {
            if (child->getFileOffset() == childOffset)
            {
                if (path)
                {
                    path->push_back (root);
                }
                return root;
            }

            if (child->isForm())
            {
                const LWO3Form* found = findParentOfOffset (
                    static_cast<const LWO3Form*> (child.get()),
                    childOffset,
                    path);
                if (found)
                {
                    if (path)
                    {
                        path->push_back (root);
                    }
                    return found;
                }
            }
        }
        return nullptr;
    }

    // Validate offset ordering
    bool validateOffsetOrdering (const LWO3Form* root, std::ostream& errors = std::cerr)
    {
        bool valid = true;
        size_t lastOffset = root->getFileOffset();

        for (const auto& child : root->getChildren())
        {
            if (child->getFileOffset() <= lastOffset)
            {
                errors << "Invalid offset ordering at 0x"
                       << std::hex << child->getFileOffset()
                       << " (previous: 0x" << lastOffset << ")\n";
                valid = false;
            }
            lastOffset = child->getFileOffset();

            if (child->isForm())
            {
                valid &= validateOffsetOrdering (
                    static_cast<const LWO3Form*> (child.get()),
                    errors);
            }
        }
        return valid;
    }

 private:
    void findElementsByIdImpl (
        const LWO3Form* form,
        uint32_t targetId,
        SearchMode mode,
        size_t startOffset,
        size_t endOffset,
        std::vector<const LWO3Element*>& currentPath,
        std::vector<SearchResult>& results)
    {
        currentPath.push_back (form);

        for (const auto& child : form->getChildren())
        {
            size_t offset = child->getFileOffset();

            // Check offset range
            if (offset < startOffset || offset > endOffset)
            {
                continue;
            }

            // Check for match
            if (child->getId() == targetId)
            {
                results.push_back ({child.get(), currentPath});
                if (mode == SearchMode::FIRST)
                {
                    return;
                }
            }

            // Recurse into forms
            if (child->isForm())
            {
                findElementsByIdImpl (
                    static_cast<const LWO3Form*> (child.get()),
                    targetId,
                    mode,
                    startOffset,
                    endOffset,
                    currentPath,
                    results);
            }
        }

        currentPath.pop_back();
    }

    void dumpHierarchyImpl (
        const LWO3Element* element,
        int depth,
        std::ostream& out)
    {
        std::string indent (depth * 2, ' ');
        out << indent << "0x" << std::hex << element->getFileOffset()
            << ": " << idToString (element->getId());

        if (!element->isForm())
        {
            const auto* chunk = static_cast<const LWO3Chunk*> (element);
            out << " (size: " << std::dec << chunk->getData().size() << ")";
        }
        out << "\n";

        if (element->isForm())
        {
            const auto* form = static_cast<const LWO3Form*> (element);
            for (const auto& child : form->getChildren())
            {
                dumpHierarchyImpl (child.get(), depth + 1, out);
            }
        }
    }
};