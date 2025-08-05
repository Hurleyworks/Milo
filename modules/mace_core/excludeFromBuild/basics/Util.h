#pragma once

// NO MORE TRACING  Yay!
#define TRACE(x)

const LEVELS TESTING{INFO.value + 1, "TESTING"};
const LEVELS CRITICAL{WARNING.value + 1, "CRTICAL"};

#if defined(_WIN32) || defined(_WIN64)
#define __FUNCTION_NAME__ __func__
#define _FN_ __FUNCTION_NAME__
#ifndef NOMINMAX
#define NOMINMAX
#endif

#undef near
#undef far
#undef RGB
#endif

#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2

// https://stackoverflow.com/questions/6942273/how-to-get-a-random-element-from-a-c-container
//  https://gist.github.com/cbsmith/5538174
template <typename RandomGenerator = std::default_random_engine>
struct random_selector
{
    // On most platforms, you probably want to use std::random_device("/dev/urandom")()
    random_selector (RandomGenerator g = RandomGenerator (std::random_device()())) :
        gen (g) {}

    template <typename Iter>
    Iter select (Iter start, Iter end)
    {
        std::uniform_int_distribution<> dis (0, std::distance (start, end) - 1);
        std::advance (start, dis (gen));
        return start;
    }

    // convenience function
    template <typename Iter>
    Iter operator() (Iter start, Iter end)
    {
        return select (start, end);
    }

    // convenience function that works on anything with a sensible begin() and end(), and returns with a ref to the value type
    template <typename Container>
    auto operator() (const Container& c) -> decltype (*begin (c))&
    {
        return *select (begin (c), end (c));
    }

 private:
    RandomGenerator gen;
};

// makes it illegal to copy a derived class
// https://github.com/heisters/libnodes
struct Noncopyable
{
 protected:
    Noncopyable() = default;
    ~Noncopyable() = default;
    Noncopyable (const Noncopyable&) = delete;
    Noncopyable& operator= (const Noncopyable&) = delete;
};

// provides derived classes with automatically assigned,
// globally unique numeric identifiers
// https://github.com/heisters/libnodes
class HasId
{
 public:
    HasId() :
        mId (++sId)
    {
        // LOG (DBUG) << mId;
    }

    ItemID id() const { return mId; }
    void setID (ItemID itemID) { mId = itemID; }

    void staticReset (int id = 0) { sId = id; }

 protected:
    static ItemID sId;
    ItemID mId;
};

// from the Code Blacksmith
// https://www.youtube.com/watch?v=GV0JMHOpoEw
class ScopedStopWatch
{
 public:
    using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                     std::chrono::high_resolution_clock,
                                     std::chrono::steady_clock>;
    ScopedStopWatch (const char function[] = "unknown function") :
        func (function)
    {
    }
    ~ScopedStopWatch()
    {
        LOG (DBUG) << "\n"
                   << func << " took " << std::chrono::duration_cast<std::chrono::milliseconds> (Clock::now() - start).count() << " milliseconds";
    }

 private:
    const char* func = nullptr;
    Clock::time_point start = Clock::now();
};

// A type-safe container for heterogeneous property values
// Maps enum or integer keys to values of arbitrary types using std::any
template <class PROPERTY>
class AnyValue
{
 public:
    using KeyType = PROPERTY;
    using ValueMap = std::unordered_map<int, std::any>;

    AnyValue() = default;
    ~AnyValue() = default;

    // Add a default value if the key doesn't already exist
    void addDefault (const PROPERTY& key, const std::any& value)
    {
        int keyValue = convertToInt (key);
        if (map_.find (keyValue) == map_.end())
        {
            map_.insert (std::make_pair (keyValue, value));
        }
    }

    // Set or update a value
    void setValue (const PROPERTY& key, const std::any& value)
    {
        int keyValue = convertToInt (key);
        auto it = map_.find (keyValue);
        if (it == map_.end())
            map_.insert (std::make_pair (keyValue, value));
        else
            it->second = value;
    }

    // Check if a key exists in the map
    bool hasKey (const PROPERTY& key) const
    {
        return map_.find (convertToInt (key)) != map_.end();
    }

    // Get a reference to a value (can throw)
    template <typename T>
    T& getRef (const PROPERTY& key)
    {
        return std::any_cast<T&> (getValue (key));
    }

    // Get a copy of the value (can throw)
    template <typename T>
    T getVal (const PROPERTY& key)
    {
        auto& value = getValue (key);
        return std::any_cast<T> (value);
    }

    // Get a value with a default if not found or wrong type
    template <typename T>
    T getValOr (const PROPERTY& key, const T& defaultValue) const
    {
        try
        {
            auto it = map_.find (convertToInt (key));
            if (it != map_.end())
            {
                return std::any_cast<T> (it->second);
            }
        }
        catch (...)
        {
            // Just return default on any exception
        }
        return defaultValue;
    }

    // Get a pointer to the value, returns nullptr on failure
    template <typename T>
    T* getPtr (const PROPERTY& key)
    {
        try
        {
            auto& val = getValue (key);
            return std::any_cast<T> (&val);
        }
        catch (...)
        {
            return nullptr;
        }
    }

    // Clear all properties
    void clear()
    {
        map_.clear();
    }

    // Get number of properties
    size_t size() const
    {
        return map_.size();
    }

 private:
    ValueMap map_;
    std::any empty_; // Empty value returned when key doesn't exist

    // Helper to convert key to integer, supporting both enum and integer keys
    template <typename T = PROPERTY>
    static int convertToInt (const T& key)
    {
        if constexpr (std::is_enum_v<T>)
        {
            return static_cast<int> (key);
        }
        else
        {
            return static_cast<int> (key);
        }
    }

    // Get a reference to a value
    std::any& getValue (const PROPERTY& key)
    {
        auto it = map_.find (convertToInt (key));
        if (it != map_.end())
            return it->second;
        return empty_;
    }

    const std::any& getValue (const PROPERTY& key) const
    {
        auto it = map_.find (convertToInt (key));
        if (it != map_.end())
            return it->second;
        return empty_;
    }
};
#if 0
// store and retrieve any type from a map
template <class PROPERTY>
class AnyValue
{
    using ValueMap = std::unordered_map<int, std::any>;

 public:
    AnyValue() = default;
    ~AnyValue() = default;

    void addDefault (const PROPERTY& key, const std::any& value) { map_.insert (std::make_pair (key, value)); }
    void setValue (const PROPERTY& key, const std::any& value)
    {
        auto it = map_.find (key);
        if (it == map_.end())
            map_.insert (std::make_pair (key, value));
        else
            it->second = value;
    }

    template <typename T>
    T& getRef (const PROPERTY& key) { return std::any_cast<T> (getValue (key)); }

    template <typename T>
    T getVal (const PROPERTY& key) { return std::any_cast<T> (getValue (key)); }

    template <typename T>
    T* getPtr (const PROPERTY& key) { return std::any_cast<T> (&getValue (key)); }

 private:
    ValueMap map_;
    std::any empty_;

    std::any& getValue (const PROPERTY& key)
    {
        auto it = map_.find (key);
        if (it != map_.end())
            return it->second;

        return empty_;
    }

}; // end class AnyValue


#endif



// Extract scene name from a folder path that might contain it
// Format example: "/path/to/sceneName_04_10_2025_03_22pm"
inline std::string extractSceneNameFromPath (const fs::path& path)
{
    // Get the last component of the path
    std::string lastComponent = path.filename().string();

    // Check if the last component contains a timestamp pattern
    size_t timestampPos = lastComponent.find ("_");

    if (timestampPos != std::string::npos)
    {
        // Check if this looks like our timestamp format (has multiple underscores)
        size_t secondUnderscorePos = lastComponent.find ("_", timestampPos + 1);
        if (secondUnderscorePos != std::string::npos &&
            secondUnderscorePos < lastComponent.length() - 1)
        {
            // Extract the scene name (everything before the first underscore)
            return lastComponent.substr (0, timestampPos);
        }
    }

    // Fallback: if there's no clear timestamp pattern,
    // use the folder name as is, or "untitled" if it seems inappropriate
    return (lastComponent.empty() || lastComponent == "." || lastComponent == "..")
               ? "untitled"
               : lastComponent;
}

// TimeUtils.h - Updated createTimestampString function with seconds (no milliseconds)

// Generate a timestamp string with seconds precision
inline std::string createTimestampString (bool lowercase = true)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t (now);

    std::tm localTime;

#ifdef _WIN32
    localtime_s (&localTime, &time);
#else
    localtime_r (&time, &localTime);
#endif

    // Format with seconds: "_%m_%d_%Y_%I_%M_%S%p"
    char timestamp[40];
    std::strftime (timestamp, sizeof (timestamp), "_%m_%d_%Y_%I_%M_%S%p", &localTime);

    std::string timestampStr (timestamp);

    if (lowercase)
    {
        // Convert AM/PM to lowercase
        std::transform (timestampStr.begin(), timestampStr.end(),
                        timestampStr.begin(), ::tolower);
    }

    return timestampStr;
}

// Generate a timestamped filename with optional frame number
inline std::string createTimestampedFilename (
    const std::string& prefix,
    const std::string& sceneName,
    uint32_t frameNumber = 0,
    const std::string& extension = ".exr")
{
    // Use "untitled" if sceneName is empty
    std::string baseSceneName = sceneName.empty() ? "untitled" : sceneName;

    // Create timestamp
    std::string timestamp = createTimestampString();

    // Ensure extension starts with a dot
    std::string ext = extension.empty() ? ".exr" : (extension[0] == '.' ? extension : "." + extension);

    // Generate filename with or without frame number
    if (frameNumber > 0)
    {
        // Include frame number
        return std::format ("{}_{}{}_{}{}",
                            prefix,
                            baseSceneName,
                            timestamp,
                            std::format ("{:05d}", frameNumber),
                            ext);
    }
    else
    {
        // No frame number
        return std::format ("{}_{}{}{}",
                            prefix,
                            baseSceneName,
                            timestamp,
                            ext);
    }
}

// Generate a complete timestamped file path by combining base path and timestamped filename
inline fs::path createTimestampedFilePath (
    const fs::path& basePath,
    const std::string& prefix,
    const std::string& sceneName,
    uint32_t frameNumber = 0,
    const std::string& extension = ".exr")
{
    std::string filename = createTimestampedFilename (prefix, sceneName, frameNumber, extension);
    return basePath / filename;
}

inline double generateRandomDouble (double lower_bound, double upper_bound)
{
    std::random_device rd;                                             // Obtain a random number from hardware
    std::mt19937 gen (rd());                                           // Seed the generator
    std::uniform_real_distribution<> distr (lower_bound, upper_bound); // Define the range

    return distr (gen); // Generate numbers
}

inline double randomUniform (double min, double max)
{
    std::random_device rd;
    std::default_random_engine generator (rd());
    std::uniform_real_distribution<double> distribution (min, max);

    return distribution (generator);
}

inline double generateRandomDouble()
{
    static std::mt19937 generator (12345);              // A Mersenne Twister pseudo-random generator with a seed of 12345
    std::uniform_real_distribution<> distr (-1.0, 1.0); // Define the distribution between -1.0 and 1.0

    return distr (generator);
}

inline std::string getFileNameWithoutExtension (const std::filesystem::path& filePath)
{
    return filePath.stem().string();
}

inline std::filesystem::path changeFileExtensionToJpeg (const std::filesystem::path& pngPath)
{
    std::filesystem::path newPath = pngPath;
    newPath.replace_extension (".jpeg");
    return newPath;
}

inline bool isValidPath (const std::filesystem::path& filePath, const std::string& rejectWord)
{
    // Convert the file path to a string
    std::string pathStr = filePath.generic_string();

    // Convert the string to lowercase for case-insensitive search
    std::transform (pathStr.begin(), pathStr.end(), pathStr.begin(),
                    [] (unsigned char c)
                    { return std::tolower (c); });

    // Convert the reject word to lowercase for case-insensitive comparison
    std::string lowerRejectWord = rejectWord;
    std::transform (lowerRejectWord.begin(), lowerRejectWord.end(), lowerRejectWord.begin(),
                    [] (unsigned char c)
                    { return std::tolower (c); });

    // Search for the reject word in the path string
    if (pathStr.find (lowerRejectWord) != std::string::npos)
    {
        // Reject word is found in the path
        return false;
    }

    // Reject word is not found in the path
    return true;
}

inline bool pathContainsIgnoreCase (const std::filesystem::path& filePath, const std::string& searchWord)
{
    // Convert the file path to a string
    std::string pathStr = filePath.generic_string();

    // Convert the string to lowercase for case-insensitive search
    std::transform (pathStr.begin(), pathStr.end(), pathStr.begin(),
                    [] (unsigned char c)
                    { return std::tolower (c); });

    // Convert the search word to lowercase for case-insensitive comparison
    std::string lowerSearchWord = searchWord;
    std::transform (lowerSearchWord.begin(), lowerSearchWord.end(), lowerSearchWord.begin(),
                    [] (unsigned char c)
                    { return std::tolower (c); });

    // Look for the search word in the path string
    if (pathStr.find (lowerSearchWord) != std::string::npos)
    {
        // word is found in the path
        return true;
    }

    // Reject word is not found in the path
    return false;
}

// rational.cpp by Bill Weinman <http://bw.org/>
// updated 2015-06-01
inline void message (const char* s)
{
    puts (s);
    fflush (stdout);
}

enum ErrorSeverity
{
    Information,
    Warning,
    Critical,
};

struct ErrMsg
{
    std::string message = "";
    ErrorSeverity severity = ErrorSeverity::Information;
};


enum class SceneState
{
    Normal,
    Clearing,
    Loading,
    Ready
};

// Helper function to format byte sizes into human readable string
inline std::string formatByteSize (size_t bytes)
{
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unitIndex = 0;
    double size = bytes;

    while (size >= 1024 && unitIndex < 3)
    {
        size /= 1024;
        unitIndex++;
    }

    // Format with 2 decimal places
    return std::format ("{:.2f} {}", size, units[unitIndex]);
}


// for libassert
inline void custom_fail (const libassert::assertion_info& assertion)
{
    LOG (CRITICAL) << assertion.to_string (libassert::terminal_width (STDERR_FILENO), libassert::color_scheme::ansi_rgb);
}
inline std::string readTxtFile (const std::filesystem::path& filepath)
{
    std::ifstream ifs;
    ifs.open (filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string (sstream.str());
}

inline std::vector<char> readBinaryFile (const std::filesystem::path& filepath)
{
    std::vector<char> ret;

    std::ifstream ifs;
    ifs.open (filepath, std::ios::in | std::ios::binary | std::ios::ate);
    if (ifs.fail())
        return std::move (ret);

    std::streamsize fileSize = ifs.tellg();
    ifs.seekg (0, std::ios::beg);

    ret.resize (fileSize);
    ifs.read (ret.data(), fileSize);

    return std::move (ret);
}

struct FileServices
{
    static void copyFiles (const std::string& searchFolder, const std::string& destFolder, const std::string& extension, bool recursive = true)
    {
        std::filesystem::recursive_directory_iterator dirIt (searchFolder), end;
        while (dirIt != end)
        {
            if (dirIt->path().extension() == extension || extension == "*")
            {
                std::filesystem::copy (dirIt->path(), destFolder + "/" + dirIt->path().filename().string());
            }
            ++dirIt;
        }
    }

    static void moveFiles (const std::string& searchFolder, const std::string& destFolder, const std::string& extension)
    {
        for (const auto& entry : std::filesystem::directory_iterator (searchFolder))
        {
            if (entry.path().extension() == extension || extension == "*")
            {
                std::filesystem::rename (entry, destFolder + "/" + entry.path().filename().string());
            }
        }
    }

    static std::vector<std::filesystem::path> findFilesWithExtension (const std::filesystem::path& searchFolder, const std::string extension)
    {
        std::vector<std::filesystem::path> matchingFiles;

        for (auto const& dir_entry : std::filesystem::directory_iterator{searchFolder})
        {
            if (dir_entry.path().extension().string() == extension)
            {
                matchingFiles.push_back (dir_entry.path());
            }
        }

        return matchingFiles;
    }

    static std::vector<std::string> getFiles (const std::filesystem::path& searchFolder, const std::string& extension, bool recursive)
    {
        std::vector<std::string> files;
        if (recursive)
        {
            for (auto it = std::filesystem::recursive_directory_iterator (searchFolder);
                 it != std::filesystem::recursive_directory_iterator(); ++it)
            {
                try
                {
                    if (it->path().extension() == extension)
                    {
                        // LOG (DBUG) << it->path().generic_string();
                        files.push_back (it->path().generic_string());
                    }
                }
                catch (const std::exception& e)
                {
                    // Handle the error, log it, or ignore it.
                    LOG (DBUG) << "Error reading file path: " << e.what();
                }
            }
        }
        else
        {
            for (auto it = std::filesystem::directory_iterator (searchFolder);
                 it != std::filesystem::directory_iterator(); ++it)
            {
                try
                {
                    LOG (DBUG) << it->path().generic_string();
                    if (it->path().extension() == extension)
                    {
                        files.push_back (it->path().generic_string());
                    }
                }
                catch (const std::exception& e)
                {
                    // Handle the error, log it, or ignore it.
                    LOG (DBUG) << "Error reading file path: " << e.what();
                }
            }
        }
        return files;
    }

    static std::vector<std::string> getFolders (const std::string& searchFolder, bool recursive = true)
    {
        std::vector<std::string> folders;
        std::filesystem::recursive_directory_iterator dirIt (searchFolder), end;
        while (dirIt != end)
        {
            if (std::filesystem::is_directory (dirIt->status()))
            {
                folders.push_back (dirIt->path().string());
            }
            ++dirIt;
        }
        return folders;
    }

    static std::vector<std::string> getTextFileLines (const std::string& filePath)
    {
        std::vector<std::string> lines;
        std::ifstream file (filePath);
        if (!file)
            return lines;

        std::string line;
        while (getline (file, line))
        {
            lines.push_back (line);
        }
        return lines;
    }

    static std::string findFilePath (const std::string& searchFolder, const std::string& fileName)
    {
        std::filesystem::recursive_directory_iterator dirIt (searchFolder), end;
        while (dirIt != end)
        {
            if (dirIt->path().filename() == fileName)
            {
                return dirIt->path().string();
            }
            ++dirIt;
        }
        return "";
    }

    static std::optional<std::filesystem::path> findFileInFolder (
        const std::filesystem::path& folder,
        const std::string& filename)
    {
        for (const auto& entry : std::filesystem::recursive_directory_iterator (folder))
        {
            if (entry.path().filename() == filename)
            {
                return entry.path();
            }
        }
        return std::nullopt;
    }
};

inline std::string getParentFolderName (const std::filesystem::path& path)
{
    return path.parent_path().filename().string();
}

inline bool hasObjExtension (const std::filesystem::path& filePath)
{
    return filePath.extension() == ".obj";
}

inline bool hasGltfExtension (const std::filesystem::path& filePath)
{
    return filePath.extension() == ".gltf";
    // return filePath.extension() == ".gltf" || filePath.extension() == ".glb";
}

inline bool isStaticBody (const std::filesystem::path& filePath)
{
    std::string filename = filePath.stem().string();

    // Check if filename starts with "static"
    return filename.rfind ("static", 0) == 0;
}

inline std::vector<std::filesystem::path> getDirectoriesInParentDirectories (const std::filesystem::path& filePath, int32_t count = 3)
{
    std::vector<std::filesystem::path> directories;

    if (!std::filesystem::is_regular_file (filePath))
    {
        throw std::invalid_argument ("The provided path is not a file.");
    }

    auto parent = filePath.parent_path();

    for (int i = 0; i < count && !parent.empty(); ++i)
    {
        // Recursive search for directories in the parent directory
        for (const auto& entry : std::filesystem::recursive_directory_iterator (parent))
        {
            if (std::filesystem::is_directory (entry.path()))
            {
                directories.push_back (entry.path());
            }
        }
        parent = parent.parent_path(); // Move to the next parent
    }

    return directories;
}

inline bool findAssetFromFileLocation (std::string& assetPath, const fs::path& startSearchFrom)
{
    std::filesystem::path path (assetPath);
    if (path.is_absolute() && std::filesystem::exists (path))
    {
        return true;
    }

    auto parents = getDirectoriesInParentDirectories (startSearchFrom);
    for (const auto& dir : parents)
    {
        std::filesystem::path testPath = std::filesystem::path (dir) / path.filename();

        if (std::filesystem::exists (testPath))
        {
            assetPath = testPath.string();
            return true;
        }
    }

    return false;
}

template <typename EnumType>
int toInt (EnumType value)
{
    return static_cast<int> (value);
}

inline std::string supportedImageFormats()
{
    return "*.png;*.bmp;*.BMP;*.jpg;*.jpeg;*.JPG;*.exr;*.hdr;*.tga;*.targa;*.tif";
}

inline bool isSupportedImageFormat (const std::string& extension)
{
    return supportedImageFormats().find (extension) != std::string::npos;
}

inline std::string supportedMeshFormats()
{
    return "*.gltf;*.GLTF;*.glb;*.GLB;*.obj;*.OBJ;*.lwo;*.LWO;*.ply;*.PLY";
}

inline std::string supportedAnimationFormats()
{
    return "*.ozz;*.OZZ";
}

inline bool isSupportedMeshFormat (const std::string& extension)
{
    return supportedMeshFormats().find (extension) != std::string::npos;
}

inline bool isSupportedAnimationFormat (const std::string& extension)
{
    return supportedAnimationFormats().find (extension) != std::string::npos;
}

inline bool isImageFile (const fs::path& filePath)
{
    static const std::vector<std::string> imageExtensions = {
        ".png", ".jpg", ".jpeg", ".tiff", ".tga", ".exr", ".hdr"};

    std::string ext = filePath.extension().string();
    std::transform (ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find (imageExtensions.begin(), imageExtensions.end(), ext) != imageExtensions.end();
}

// Finds the texture folder starting from a given model path
// Returns empty path if no texture folder is found
inline fs::path findTextureFolder (const fs::path& modelPath)
{
    if (!fs::exists (modelPath))
        return fs::path();

    // Find images root folder by walking up from model path
    fs::path searchPath = modelPath.parent_path();
    fs::path imagesFolder;

    while (searchPath.has_parent_path())
    {
        fs::path testPath = searchPath / "images";
        if (fs::exists (testPath) && fs::is_directory (testPath))
        {
            imagesFolder = testPath;
            break;
        }
        searchPath = searchPath.parent_path();
    }

    if (imagesFolder.empty())
    {
        LOG (WARNING) << "Could not find images folder for " << modelPath;
        return fs::path();
    }

    // Search recursively for folder containing textures
    try
    {
        for (const auto& entry : fs::recursive_directory_iterator (imagesFolder))
        {
            if (entry.is_regular_file() && isImageFile (entry.path()))
            {
                fs::path textureFolder = entry.path().parent_path();
                LOG (DBUG) << "Found textures in: " << textureFolder;
                return textureFolder;
            }
        }
    }
    catch (const fs::filesystem_error& e)
    {
        LOG (WARNING) << "Error searching texture folder: " << e.what();
        return fs::path();
    }

    LOG (WARNING) << "No texture files found in " << imagesFolder;
    return fs::path();
}

struct TextureKind
{
    enum class Format
    {
        Unknown,
        Single,                     // Single channel texture (e.g. roughness, metallic)
        RGB,                        // Regular RGB texture (e.g. basecolor)
        RoughnessMetallic,          // Packed roughness (G) and metallic (B)
        OcclusionRoughnessMetallic, // Packed ORM texture
        NormalMap,                  // Normal map
        HeightMap,                  // Height/bump map
        EmissiveMap,                // Emissive texture
        AmbientOcclusion            // Ambient occlusion texture
    };

    struct Channel
    {
        std::string name;
        std::string channelOutput; // "Red", "Green", "Blue", "Alpha", "Color"
        std::string bsdfInput;     // The input name on the BSDF node
    };

    Format format = Format::Unknown;
    std::vector<Channel> channels;
    bool isNormalMap = false;
    std::string debugInfo; // Stores analysis reasoning for logging
};

class TextureAnalyzer
{
 public:
    TextureKind analyzeTexture (const std::string& path, const std::string& requestedInput)
    {
        TextureKind result;
        std::string lowerPath = path;
        std::transform (lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);

        result.debugInfo = "Analyzing texture: " + path + "\n";
        result.debugInfo += "Requested input: " + requestedInput + "\n";

        // First check for normal maps
        if (isNormalMap (lowerPath))
        {
            result.format = TextureKind::Format::NormalMap;
            result.isNormalMap = true;
            result.channels.push_back ({"Normal", "Color", requestedInput});
            result.debugInfo += "Detected as: Normal Map\n";
            //LOG (DBUG) << result.debugInfo;
            return result;
        }

        // Check for packed textures
        if (auto packedFormat = detectPackedFormat (lowerPath))
        {
            result.format = *packedFormat;
            setupPackedChannels (result, *packedFormat);
            result.debugInfo += "Detected as: Packed Texture (" + formatToString (*packedFormat) + ")\n";
           // LOG (DBUG) << result.debugInfo;
            return result;
        }

        // Check for single channel textures
        if (auto singleChannel = detectSingleChannel (lowerPath, requestedInput))
        {
            result.format = TextureKind::Format::Single;
            result.channels = *singleChannel;
            result.debugInfo += "Detected as: Single Channel Texture\n";
           // LOG (DBUG) << result.debugInfo;
            return result;
        }

        // Default to RGB for base color and emission
        if (requestedInput == "Color" || requestedInput == "Luminous Color")
        {
            result.format = TextureKind::Format::RGB;
            result.channels.push_back ({"Color", "Color", requestedInput});
            result.debugInfo += "Detected as: RGB Texture\n";
           // LOG (DBUG) << result.debugInfo;
            return result;
        }

        // Unknown format - fallback to red channel
        result.format = TextureKind::Format::Unknown;
        result.channels.push_back ({"Unknown", "Red", requestedInput});
        result.debugInfo += "WARNING: Unable to determine format, defaulting to Red channel\n";
       // LOG (WARNING) << result.debugInfo;
        return result;
    }

 private:
    bool isNormalMap (const std::string& path)
    {
        return path.find ("_normal") != std::string::npos ||
               path.find ("_nor.") != std::string::npos ||
               path.find ("_nor_") != std::string::npos ||
               path.find ("_nrm.") != std::string::npos ||
               path.find ("_nrm_") != std::string::npos;
    }

    std::optional<TextureKind::Format> detectPackedFormat (const std::string& path)
    {
        std::string lowerPath = path;
        std::transform (lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);

       // LOG (DBUG) << "Analyzing packed format for path: " << path;
       // LOG (DBUG) << "Lowercase path: " << lowerPath;

        // Common glTF and DCC tool naming patterns
        const std::vector<std::string> metallicRoughnessPatterns = {
            "metalroughness",    // Standard glTF
            "metallicroughness", // Compound
            "roughnessmetallic", // Reversed compound
            "metallic_roughness",
            "roughness_metallic",
            "metallic-roughness",
            "roughness-metallic",
            "metal_rough", // Common shorthand
            "metalrough",  // Shorthand compound
            "_mrm",        // Metallic roughness map abbreviation
            "_mr"          // Shorter abbreviation
        };

        // First check for ORM patterns
        const std::vector<std::string> ormPatterns = {
            "_orm", // Occlusion Roughness Metallic
            "_orm.",
            "_arm", // Ambient Roughness Metallic
            "_rma", // Roughness Metallic Ambient
            "_mro"  // Metallic Roughness Occlusion
        };

        for (const auto& pattern : ormPatterns)
        {
            if (lowerPath.find (pattern) != std::string::npos)
            {
               // LOG (DBUG) << "Found ORM pattern: " << pattern;
                return TextureKind::Format::OcclusionRoughnessMetallic;
            }
        }

        // Then check metallic-roughness patterns
        for (const auto& pattern : metallicRoughnessPatterns)
        {
            if (lowerPath.find (pattern) != std::string::npos)
            {
               // LOG (DBUG) << "Found metallic-roughness pattern: " << pattern;
                return TextureKind::Format::RoughnessMetallic;
            }
        }

        // Special case for camelCase naming like "metalRoughness"
        if ((lowerPath.find ("metal") != std::string::npos &&
             lowerPath.find ("rough") != std::string::npos) ||
            (lowerPath.find ("metallic") != std::string::npos &&
             lowerPath.find ("rough") != std::string::npos))
        {
           // LOG (DBUG) << "Found metallic-roughness through component detection";
            return TextureKind::Format::RoughnessMetallic;
        }

       // LOG (DBUG) << "No packed format detected";
        return std::nullopt;
    }

    std::optional<std::vector<TextureKind::Channel>> detectSingleChannel (
        const std::string& path, const std::string& requestedInput)
    {
        std::vector<TextureKind::Channel> channels;

        // Define patterns for single channel textures
        struct Pattern
        {
            std::string suffix;
            std::string channel;
            std::string input;
        };

        const std::vector<Pattern> patterns = {
            {"_rough", "Red", "Roughness"},
            {"roughness", "Red", "Roughness"},
            {"_metal", "Red", "Metallic"},
            {"metallic", "Red", "Metallic"},
            {"_height", "Red", "Height"},
            {"_bump", "Red", "Height"},
            {"_ao", "Red", "Occlusion"},
            {"_ambientocclusion", "Red", "Occlusion"},
            {"_opacity", "Red", "Opacity"},
            {"_alpha", "Red", "Opacity"}};

        for (const auto& pattern : patterns)
        {
            if (path.find (pattern.suffix) != std::string::npos)
            {
                channels.push_back ({pattern.input, pattern.channel, pattern.input});
                return channels;
            }
        }

        return std::nullopt;
    }

    void setupPackedChannels (TextureKind& result, TextureKind::Format format)
    {
        switch (format)
        {
            case TextureKind::Format::OcclusionRoughnessMetallic:
                result.channels = {
                    {"Occlusion", "Red", "Occlusion"},
                    {"Roughness", "Green", "Roughness"},
                    {"Metallic", "Blue", "Metallic"}};
                break;

            case TextureKind::Format::RoughnessMetallic:
                result.channels = {
                    {"Roughness", "Green", "Roughness"},
                    {"Metallic", "Blue", "Metallic"}};
                break;

            default:
                break;
        }
    }

    std::string formatToString (TextureKind::Format format)
    {
        switch (format)
        {
            case TextureKind::Format::Single:
                return "Single";
            case TextureKind::Format::RGB:
                return "RGB";
            case TextureKind::Format::RoughnessMetallic:
                return "RoughnessMetallic";
            case TextureKind::Format::OcclusionRoughnessMetallic:
                return "ORM";
            case TextureKind::Format::NormalMap:
                return "NormalMap";
            case TextureKind::Format::HeightMap:
                return "HeightMap";
            case TextureKind::Format::EmissiveMap:
                return "EmissiveMap";
            case TextureKind::Format::AmbientOcclusion:
                return "AmbientOcclusion";
            default:
                return "Unknown";
        }
    }
};
// Fin
namespace mace
{

    inline float luminance (const Eigen::Vector3f& v)
    {
        return v.dot (Eigen::Vector3f (0.2126f, 0.7152f, 0.0722f));
    }

    inline Eigen::Vector3f reinhard_jodie (const Eigen::Vector3f& v)
    {
        float l = luminance (v);
        Eigen::Vector3f tv = v.array() / (1.0f + v.array());
        return v.array() / (1.0f + l) * (1.0f - tv.array()) + tv.array() * tv.array();
    }

    inline Eigen::Vector3f tonemap (const Eigen::Vector3f& v)
    {
        return reinhard_jodie (v);
    }

    inline Eigen::Vector3f adjust (const Eigen::Vector3f& v)
    {
        float contrast = 0.025;
        float brightness = 0;
        float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
        float r = std::max (0.0f, (v.x() - 0.5f) * contrastFactor + 0.5f + brightness);
        float g = std::max (0.0f, (v.y() - 0.5f) * contrastFactor + 0.5f + brightness);
        float b = std::max (0.0f, (v.z() - 0.5f) * contrastFactor + 0.5f + brightness);
        return Eigen::Vector3f (r, g, b);
    }

    inline Eigen::Vector3f gamma_correct (const Eigen::Vector3f& v)
    {
        float r = 1.0f / 2.2;
        return Eigen::Vector3f (std::pow (v.x(), r), std::pow (v.y(), r), std::pow (v.z(), r));
    }

    // I think this is from InstantMeshes
    inline bool atomicCompareAndExchange (volatile uint32_t* v, uint32_t newValue, uint32_t oldValue)
    {
#if defined(_WIN32)
        return _InterlockedCompareExchange (
                   reinterpret_cast<volatile long*> (v), (long)newValue, (long)oldValue) == (long)oldValue;
#else
        return __sync_bool_compare_and_swap (v, oldValue, newValue);
#endif
    }

    inline uint32_t atomicAdd (volatile uint32_t* dst, uint32_t delta)
    {
#if defined(_MSC_VER)
        return _InterlockedExchangeAdd (reinterpret_cast<volatile long*> (dst), delta) + delta;
#else
        return __sync_add_and_fetch (dst, delta);
#endif
    }

    inline float atomicAdd (volatile float* dst, float delta)
    {
        union bits
        {
            float f;
            uint32_t i;
        };
        bits oldVal, newVal;
        do
        {
#if defined(__i386__) || defined(__amd64__)
            __asm__ __volatile__ ("pause\n");
#endif
            oldVal.f = *dst;
            newVal.f = oldVal.f + delta;
        } while (!atomicCompareAndExchange ((volatile uint32_t*)dst, newVal.i, oldVal.i));
        return newVal.f;
    }

    // Byte-swapping functions
    inline uint16_t swap16 (uint16_t value)
    {
        return (value << 8) | (value >> 8);
    }

    inline uint32_t swap32 (uint32_t value)
    {
        return ((value << 24) & 0xFF000000) |
               ((value << 8) & 0x00FF0000) |
               ((value >> 8) & 0x0000FF00) |
               ((value >> 24) & 0x000000FF);
    }

    inline float swapFloat (float value)
    {
        union
        {
            float f;
            uint32_t i;
        } u;
        u.f = value;
        u.i = swap32 (u.i);
        return u.f;
    }

    inline uint64_t swap64 (uint64_t value)
    {
        return ((value & 0x00000000000000FFULL) << 56) |
               ((value & 0x000000000000FF00ULL) << 40) |
               ((value & 0x0000000000FF0000ULL) << 24) |
               ((value & 0x00000000FF000000ULL) << 8) |
               ((value & 0x000000FF00000000ULL) >> 8) |
               ((value & 0x0000FF0000000000ULL) >> 24) |
               ((value & 0x00FF000000000000ULL) >> 40) |
               ((value & 0xFF00000000000000ULL) >> 56);
    }

    inline double swapDouble (double value)
    {
        union
        {
            double d;
            uint64_t i;
        } u;
        u.d = value;
        u.i = swap64 (u.i);
        return u.d;
    }

    // readVX: Reads a variable-length index from binary data
    // For values < 0xFF00: Returns 2-byte index
    // For values >= 0xFF00: Returns 4-byte index with high byte masked
    inline uint32_t readVX (BinaryReader& reader)
    {
        uint8_t firstByte = reader.ReadUint8();

        if (firstByte == 0xFF)
        {
            // Need 3 more bytes for 4-byte form
            uint8_t byte2 = reader.ReadUint8();
            uint8_t byte3 = reader.ReadUint8();
            uint8_t byte4 = reader.ReadUint8();
            // Combine bytes in big-endian order (no need for mace::swap32
            // since we're already reading in correct order)
            return (byte2 << 16) | (byte3 << 8) | byte4;
        }
        else
        {
            // Need 1 more byte for 2-byte form
            uint8_t byte2 = reader.ReadUint8();
            // Combine bytes in big-endian order (no need for mace::swap16
            // since we're already reading in correct order)
            return (firstByte << 8) | byte2;
        }
    }

} // namespace mace