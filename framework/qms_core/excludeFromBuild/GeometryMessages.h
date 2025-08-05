#pragma once

using mace::InputEvent;
using sabi::MeshOptions;

struct loadModel
{
    loadModel (const std::filesystem::path& path, MeshOptions options = MeshOptions()) :
        filePath (path),
        meshOptions (options)
    {
    }

    QmsID id = QmsID::LoadModel;
    QmsID realID = QmsID::LoadModel;

    std::filesystem::path filePath;
    MeshOptions meshOptions;
};

struct loadModelList
{
    loadModelList (const PathList& paths, MeshOptions options = MeshOptions()) :
        modelPaths (paths),
        meshOptions(options)
    {
    }

    QmsID id = QmsID::LoadModelList;
    QmsID realID = QmsID::LoadModelList;

    PathList modelPaths;
    MeshOptions meshOptions;
};
