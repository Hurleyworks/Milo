#pragma once


struct initImageServices
{
    initImageServices (ImageCacheHandlerPtr cache) :
        imageCache (cache)
    {
    }
    QmsID id = QmsID::InitImageServices;
    QmsID realID = QmsID::InitImageServices;

    ImageCacheHandlerPtr imageCache = nullptr;
};

struct loadImageFolder
{
    loadImageFolder (const std::filesystem::path& imageFolder) :
        imageFolder (imageFolder)
    {
    }

    QmsID id = QmsID::LoadImageFolder;
    QmsID realID = QmsID::LoadImageFolder;

    std::filesystem::path imageFolder;
};

struct loadImageList
{
    loadImageList (const PathList& paths) :
        imagePaths (paths)
    {
    }

    QmsID id = QmsID::LoadImageList;
    QmsID realID = QmsID::LoadImageList;

    PathList imagePaths;
};

struct loadImage
{
    loadImage (const std::filesystem::path& path) :
        imagePath (path)
    {
    }

    QmsID id = QmsID::LoadImage;
    QmsID realID = QmsID::LoadImage;

    std::filesystem::path imagePath;
};
