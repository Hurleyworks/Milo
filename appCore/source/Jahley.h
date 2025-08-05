//
// Note:	this file is to be included in client applications ONLY
//			NEVER include this file anywhere in the engine codebase
//
#pragma once

#include "jahley/AppConfig.h"
#include "jahley/core/App.h"
#include "jahley/EntryPoint.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Repository name for searching in the path
const std::string REPOSITORY_NAME = "Milo";

// not possible to get executablePath from std::filesystem
// https://stackoverflow.com/questions/55982441/get-executables-path-with-stdfilesystem
inline std::string getExecutablePath()
{
    char path[FILENAME_MAX];
#ifdef _WIN32
    GetModuleFileNameA (NULL, path, FILENAME_MAX);
#endif

    return std::string (path);
}

// Function to get the path for external content folder
inline std::string getExternalContentFolder()
{
    std::string exePath = getExecutablePath();
    std::filesystem::path pathObj (exePath);

    std::string exefullPath = pathObj.string();
    std::size_t pos = exefullPath.rfind (REPOSITORY_NAME);
    if (pos == std::string::npos)
    {
        // REPOSITORY_NAME not found in path
        return "";
    }
    std::string pathToRepro = exefullPath.substr (0, pos + std::string (REPOSITORY_NAME).length());

    std::filesystem::path codeFolder = std::filesystem::path (pathToRepro).parent_path();
    std::filesystem::path rootFolder = codeFolder.parent_path();

    // Convert path to string
    std::string finalPath = rootFolder.string();

    // Replace all backslashes with forward slashes
    std::replace (finalPath.begin(), finalPath.end(), '\\', '/');

    // Append common_content ensuring only one slash after drive letter
    if (finalPath.back() == '/')
    {
        finalPath += "common_content/";
    }
    else
    {
        finalPath += "/common_content/";
    }

    return finalPath;
}

// Function to get the path for common content folder
inline std::string getCommonContentFolder()
{
    // Get the current working directory
    std::filesystem::path exePath = std::filesystem::current_path();
    std::string fullPath = exePath.string();

    // Find the position of the repository name in the path
    std::string::size_type pos = fullPath.rfind (REPOSITORY_NAME);

    // Extract the path up to the repository
    std::string pathToRepro = fullPath.substr (0, pos + std::string (REPOSITORY_NAME).length());

    // Create the common folder path string
    std::string commonFolder = pathToRepro + "/resources/Common/";

    // Create the folder if it doesn't exist
    std::filesystem::create_directories (commonFolder);

    return commonFolder;
}

// Function to get the resource path for a given application name
inline std::string getResourcePath (const std::string& appName)
{
    // Get the current working directory
    std::filesystem::path exePath = std::filesystem::current_path();
    std::string fullPath = exePath.string();

    // Find the position of the repository name in the path
    std::string::size_type pos = fullPath.rfind (REPOSITORY_NAME);

    // Extract the path up to the repository
    std::string pathToRepro = fullPath.substr (0, pos + std::string (REPOSITORY_NAME).length());

    // Create the resource folder path string
    std::string resourceFolder = pathToRepro + "/resources/" + appName;

    // Create the folder if it doesn't exist
    std::filesystem::create_directories (resourceFolder);

    // Return the resource folder path
    return resourceFolder;
}

// Function to get the path to the repository
inline std::string getRepositoryPath (const std::string& appName)
{
    // Get the current working directory
    std::filesystem::path exePath = std::filesystem::current_path();
    std::string fullPath = exePath.string();

    // Find the position of the repository name in the path
    std::string::size_type pos = fullPath.rfind (REPOSITORY_NAME);

    // Extract the path up to the repository
    std::string pathToRepro = fullPath.substr (0, pos + std::string (REPOSITORY_NAME).length());

    // Return the repository path
    return pathToRepro;
}



 