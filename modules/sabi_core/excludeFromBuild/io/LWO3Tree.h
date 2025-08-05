#pragma once

// LWO3Tree: A class for reading LightWave Object (LWO) files in the LWO3 format.
//
// This class provides functionality to parse LWO3 files and construct a hierarchical
// representation of the file's structure using LWO3Form and LWO3Chunk objects.
// It handles the reading of FORM and chunk elements, properly interpreting their
// identifiers and sizes, and organizes them into a tree-like structure.
//
// Key features:
// - Reads and validates LWO3 file headers
// - Parses nested FORM structures and individual chunks
// - Constructs a hierarchical representation of the LWO3 file content
// - Handles byte-swapping for cross-platform compatibility
// - Ensures proper alignment by skipping padding bytes when necessary
//
// Usage:
//   LWO3Tree reader;
//   std::unique_ptr<LWO3Form> rootForm = reader.read("path/to/file.lwo");
//
// Note: This class assumes that the input file is a valid LWO3 file. It performs
// basic validation on the file header but relies on correct internal structure
// for successful parsing.


class LWO3Tree
{
 public:
    // Default constructor
    LWO3Tree() = default;

    // Default destructor
    ~LWO3Tree() = default;

    // Reads an LWO3 file and returns the root LWO3Form
    // Parameters:
    //   lwoPath: The file system path to the LWO3 file
    // Returns:
    //   A shared_ptr to the root LWO3Form containing the entire file structure,
    //   or nullptr if the file is invalid or cannot be read
    std::shared_ptr<LWO3Form> read (const fs::path& lwoPath);

 private:
    // Recursively reads and constructs the hierarchical structure of FORM and chunk elements
    // Parameters:
    //   reader: The BinaryReader object positioned at the start of an element
    //   parent: Pointer to the parent LWO3Form to which the read element will be added
    // This method handles both FORM (which may contain nested elements) and chunk elements
    void readElement (BinaryReader& reader, LWO3Form* parent);
};