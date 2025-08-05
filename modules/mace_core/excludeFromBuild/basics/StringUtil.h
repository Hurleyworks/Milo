#pragma once

// from MaterialX f
/// A vector of strings.
using StringVec = std::vector<std::string>;
/// An unordered map with strings as both keys and values.
using StringMap = std::unordered_map<std::string, std::string>;
/// A set of strings.
using StringSet = std::set<std::string>;


// convert string into hex value
template <typename T>
inline T string2HexVal (const std::string& s)
{
    T value;
    std::stringstream ss;
    ss << s;
    ss >> std::hex >> value;

    return value;
}

// convert built-in type into a string
template <class T>
inline std::string ToString (T value)
{
    std::ostringstream buffer;
    buffer.setf (std::ios::fixed, std::ios::floatfield);
    buffer.precision (8);

    buffer << value;

    return buffer.str().c_str();
}

// convert string to a built-in type
template <typename T>
inline T StringToValue (const std::string& s)
{
    T value;
    std::stringstream ss;
    ss << s;
    ss >> value;

    return value;
}

// TableLookup
inline int TableLookup (const char* str, const char* table[], unsigned int count)
{
    const unsigned int invalid = count;
    if (!str)
        return invalid;
    while (count--)
        if (!strcmp (table[count], str))
            return count;

    return invalid;
}

inline void vecStr3f (const Eigen::Vector3f& v, LEVELS level, const char function[] = "unknown function")
{
    LOG (level) << "--- called from " << function;

    std::ostringstream buffer;
    buffer.setf (std::ios::fixed, std::ios::floatfield);
    buffer.precision (6);
    buffer << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";

    LOG (level) << buffer.str();
}

inline void vecStr3d (const Eigen::Vector3d& v, LEVELS level, const char function[] = "unknown function")
{
    LOG (level) << "--- called from " << function;

    std::ostringstream buffer;
    buffer.setf (std::ios::fixed, std::ios::floatfield);
    buffer.precision (8);
    buffer << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";

    LOG (level) << buffer.str();
}

inline void vecStr4f (const Eigen::Vector4f& v, LEVELS level, const char function[] = "unknown function")
{
    std::ostringstream buffer;
    buffer.setf (std::ios::fixed, std::ios::floatfield);
    buffer.precision (6);
    buffer << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ")";

    LOG (level) << buffer.str();
}

inline void matStr4f (const Eigen::Affine3f& t, LEVELS level, const char function[] = "unknown function")
{
    LOG (level) << "--- called from " << function;

    for (int i = 0; i < 4; i++)
        vecStr4f (t.matrix().row (i), level, function);
}

inline void matStr4f (const Eigen::Matrix4f& m, LEVELS level, const char function[] = "unknown function")
{
    LOG (level) << "--- called from " << function;

    for (int i = 0; i < 4; i++)
        vecStr4f (m.row (i), level, function);
}

inline void matStr3f (const Eigen::Matrix3f& m, LEVELS level, const char function[] = "unknown function")
{
    LOG (level) << "--- called from " << function;

    std::ostringstream buffer;
    buffer.setf (std::ios::fixed, std::ios::floatfield);
    buffer.precision (6);

    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3f v = m.row (i);
        buffer << "\n\t\t|" << v[0] << ", " << v[1] << ", " << v[2] << "|";
    }

    LOG (level) << buffer.str();
}

inline void bboxStrf (const Eigen::AlignedBox3f b, LEVELS level, const char function[] = "unknown function")
{
    LOG (level) << "--- called from " << function;

    std::ostringstream buffer;
    buffer.setf (std::ios::fixed, std::ios::floatfield);
    buffer.precision (6);
    buffer << "( min " << b.min()[0] << ", " << b.min()[1] << ", " << b.min()[2] << " max " << b.max()[0] << ", " << b.max()[1] << ", " << b.max()[2] << ")";

    LOG (level) << buffer.str();
}

// Function to check if 'str' starts with 'prefix'
inline bool startsWith (const std::string& str, const std::string& prefix)
{
    // If prefix is empty, return true
    if (prefix.empty())
    {
        return true;
    }

    // If prefix is longer than str, it can't be a prefix
    if (prefix.size() > str.size())
    {
        return false;
    }

    // Compare the substring of str with prefix
    return std::equal (prefix.begin(), prefix.end(), str.begin());
}
