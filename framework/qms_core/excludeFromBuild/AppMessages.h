#pragma once

using mace::InputEvent;
using sabi::CameraHandle;

struct initializeApp
{
    initializeApp (const CameraHandle& camera) :
        camera (camera)
    {
    }
    QmsID id = QmsID::InitializeApp;
    QmsID realID = QmsID::InitializeApp;

    CameraHandle camera = nullptr;
};

struct refreshApp
{
    refreshApp (InputEvent& input) :
        input (input)
    {
    }
    InputEvent input;
    QmsID id = QmsID::RefreshApp;
    QmsID realID = QmsID::RefreshApp;
};