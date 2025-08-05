//#include "LWO3Navigator.h"
//#include "LWO3MaterialManager.h"

LWO3MaterialManager::LWO3MaterialManager (std::shared_ptr<LWO3Form> root)
{
    if (root)
    {
        extractMaterials (root);
    }
    else
    {
        LOG (WARNING) << "Null root form provided to LWO3MaterialManager";
    }
}
void LWO3MaterialManager::extractMaterials (std::shared_ptr<LWO3Form> root)
{
    LWO3Navigator nav;
    auto results = nav.findElementsById (root.get(), LWO::SURF);

    for (const auto& result : results)
    {
        if (!result.element->isForm()) continue;

        const auto* surfForm = static_cast<const LWO3Form*> (result.element);

        // Find surface name in ANON chunk
        for (const auto& child : surfForm->getChildren())
        {
            if (!child->isForm() && child->getId() == LWO::ANON)
            {
                const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                BinaryReader reader (
                    const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                    static_cast<uint32_t> (chunk->getData().size()));

                std::string surfaceName = reader.ReadNullTerminatedString();
                if (!surfaceName.empty())
                {
                    materials_.push_back (std::make_shared<LWO3Material> (surfaceName, root, surfForm));
                }
            }
        }
    }
}

const LWO3Material* LWO3MaterialManager::getMaterial (const std::string& surfaceName) const
{
    auto it = std::find_if (materials_.begin(), materials_.end(),
                            [&] (const auto& mat)
                            { return mat->getName() == surfaceName; });
    return it != materials_.end() ? it->get() : nullptr;
}