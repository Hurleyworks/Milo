#pragma once

using sabi::RenderableList;
using sabi::RenderableNode;
using sabi::SelectionOptions;

struct addNodeToWorld
{
    addNodeToWorld (RenderableNode& renderable) :
        node (renderable)
    {
    }

    QmsID id = QmsID::AddNodeToWorld;
    QmsID realID = QmsID::AddNodeToWorld;

    RenderableNode node = nullptr;
};

struct addNodeListToWorld
{
    addNodeListToWorld (RenderableList&& nodeList) :
        nodeList (std::move (nodeList))
    {
    }

    QmsID id = QmsID::AddNodeListToWorld;
    QmsID realID = QmsID::AddNodeListToWorld;

    RenderableList nodeList;
};

struct createInstanceStack
{
    createInstanceStack (uint32_t count) :
        count (count)
    {
    }

    QmsID id = QmsID::CreateInstanceStack;
    QmsID realID = QmsID::CreateInstanceStack;
    uint32_t count = 0;
};

struct createInstanceClump
{
    createInstanceClump (uint32_t count) :
        count (count)
    {
    }

    QmsID id = QmsID::CreateInstanceClump;
    QmsID realID = QmsID::CreateInstanceClump;
    uint32_t count = 0;
};

struct selectAll
{
    QmsID id = QmsID::SelectAll;
    QmsID realID = QmsID::SelectAll;
};

struct deselectAll
{
    QmsID id = QmsID::DeselectAll;
    QmsID realID = QmsID::DeselectAll;
};

struct selectByOptions
{
    selectByOptions (SelectionOptions options) :
        options(options)
    {
    }

    QmsID id = QmsID::SelectByOptions;
    QmsID realID = QmsID::SelectByOptions;

    SelectionOptions options;
};