//#include "LWO3Tree.h"

std::shared_ptr<LWO3Form> LWO3Tree::read (const fs::path& lwoPath)
{
    BinaryReader reader (lwoPath.string().c_str());

    // Read the FORM identifier
    uint32_t formId = reader.ReadUint32();
    if (formId != mace::swap32 (LWO::FORM))
    {
        LOG (WARNING) << "Invalid LWO3 file: FORM identifier not found";
        return nullptr;
    }

    // Read the file size (we don't need to use this, but we need to skip it)
    reader.Skip (4);

    // Read the LWO3 identifier
    uint32_t lwo3Id = reader.ReadUint32();
    if (lwo3Id != mace::swap32 (LWO::LWO3))
    {
        LOG (WARNING) << "Invalid LWO3 file: LWO3 identifier not found";
        return nullptr;
    }

    auto rootForm = std::make_shared<LWO3Form> (LWO::LWO3);

    while (reader.Position() < reader.Length())
    {
        readElement (reader, rootForm.get());
    }

    return rootForm;
}

// Reads a chunk or form element at the current position and populates the parent
void LWO3Tree::readElement (BinaryReader& reader, LWO3Form* parent)
{
    size_t elementOffset = reader.Position();
    uint32_t id = reader.ReadUint32();
    uint32_t size = mace::swap32 (reader.ReadUint32());

    if (id == mace::swap32 (LWO::FORM))
    {
        uint32_t formType = reader.ReadUint32();
        auto form = std::make_unique<LWO3Form> (mace::swap32 (formType), elementOffset);

        size_t endPosition = reader.Position() + size - 4;
        while (reader.Position() < endPosition)
        {
            readElement (reader, form.get());
        }

        parent->addChild (std::move (form));
    }
    else
    {
        auto chunk = std::make_unique<LWO3Chunk> (mace::swap32 (id), elementOffset);
        std::vector<uint8_t> data (size);
        reader.ReadToMemory (data.data(), size);
        chunk->setData (std::move (data));
        parent->addChild (std::move (chunk));

        if (size % 2 != 0)
        {
            reader.Skip (1);
        }
    }
}
