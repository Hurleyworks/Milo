#include "Jahley.h"
#include "View.h"
#include "Standard.h"
#include "CommandProcessor.h"
#include "Model.h"
#include "Controller.h"

const std::string APP_NAME = "RenderDog";

using nanogui::Vector2i;
using nanogui::Vector4i;

class Application : public Jahley::App
{
 public:
    Application (DesktopWindowSettings settings = DesktopWindowSettings(), bool windowApp = false) :
        Jahley::App (settings, windowApp)
    {
        properties.init();

        view = dynamic_cast<View*> (window->getScreen());

        // Initialize the command processor
        commandProcessor = std::make_unique<CommandProcessor> (view, properties);

        // Connect CommandProcessor signals to Model methods
        commandProcessor->hdrImageChangeEmitter.connect<Model, &Model::addSkyDomeImage> (&model);
        commandProcessor->loadGltfEmitter.connect<Model, &Model::loadGLTF> (&model);
        commandProcessor->loadGltfFolderEmitter.connect<Model, &Model::loadGLTF> (&model);

        // view to model connections
        view->onDrop.connect<Model, &Model::onDrop> (&model);
        view->onPipelineChange.connect<Model, &Model::setPipeline> (&model);
        view->onEnablePipelineSystem.connect<Model, &Model::enablePipelineSystem> (&model);
        view->onEngineChange.connect<Model, &Model::setEngine> (&model);

        // view to controller connections for environment controls
        view->onEnvironmentIntensityChange.connect<Controller, &Controller::onEnvironmentIntensityChange> (&controller);
        view->onEnvironmentRotationChange.connect<Controller, &Controller::onEnvironmentRotationChange> (&controller);

        // view to controller connection for animation controls
        view->onAnimationToggle.connect<Controller, &Controller::onAnimationToggle> (&controller);

        // view to controller connections for area light controls
        view->onAreaLightIntensityChange.connect<Controller, &Controller::onAreaLightIntensityChange> (&controller);
        view->onAreaLightEnable.connect<Controller, &Controller::onAreaLightEnable> (&controller);

        // view to model connection for RiPR render mode changes
        view->onRiPRRenderModeChange.connect<Model, &Model::setRiPRRenderMode> (&model);

        // model to view connection for HDR loading
        model.hdrLoadedEmitter.connect<Application, &Application::onHDRLoaded> (this);

        view->getCanvas()->inputEmitter.connect<&App::onInputEvent> (*this);

        std::string resourceFolder = getResourcePath (APP_NAME);
        properties.renderProps->setValue (RenderKey::ResourceFolder, resourceFolder);

        std::string repoFolder = getRepositoryPath (APP_NAME);
        properties.renderProps->setValue (RenderKey::RepoFolder, repoFolder);

        std::string commonFolder = getCommonContentFolder();
        properties.renderProps->setValue (RenderKey::CommonFolder, commonFolder);

        std::string externalContent = getExternalContentFolder();
        properties.renderProps->setValue (RenderKey::ExternalContentFolder, externalContent);

        // framegrabs are stored in the resource/screenshot folder
        properties.renderProps->setValue (RenderKey::FramegrabFolder, resourceFolder + "/screenshots");

        //  build configuration for CUDA kernel compilation strategy
        properties.renderProps->setValue (RenderKey::SoftwareReleaseMode, true);
        properties.renderProps->setValue (RenderKey::UseEmbeddedPTX, true);

        std::string contentFolder = "E:/common_content/models";
        properties.renderProps->setValue (RenderKey::ContentFolder, contentFolder);

        properties.renderProps->setValue (RenderKey::RenderPasses, 2u);
        properties.renderProps->setValue (RenderKey::RenderScale, PreviewScaleFactor::x1);

        properties.renderProps->setValue (RenderKey::UseEmbeddedPTX, true);
    }

    void onInit() override
    {
        view->debug();

        model.initialize (view->getCamera(), properties);
        controller.initialize (properties, view->getCamera());

        // Enable the rendering engine system immediately (not using legacy pipelines)
        model.enablePipelineSystem (true);

        // Create the socket server
        socketServer = std::make_unique<ActiveSocketServer>();

        // Initialize the server on default port (9875)
        socketServer->getMessenger().send (QMS::initSocketServer (9875, messengers));

        fs::path hdr = "E:/common_content/RiPR_demo_content/HDRI/lakeside_sunrise_4k.hdr";
        // hdr = "C:/common_content/HDRI/cape_hill_4k.hdr";
        model.addSkyDomeImage (hdr);

        std::string testModel = "E:/common_content/models/test_model_for_gltf/scene.gltf";
        std::string helmet = "E:/common_content/models/DamagedHelmet/glTF/DamagedHelmet.gltf";
        std::string cash = "E:/common_content/models/CashRegister_01_4k/CashRegister_01_4k.gltf";
        std::string scifi = "E:/common_content/models/SciFiHelmet/glTF/SciFiHelmet.gltf";
        std::string turtle = "E:/common_content/models/turtle/scene.gltf";
        std::string bust = "E:/common_content/models/bust_bohuslav_martinu/scene.gltf";
        std::string phone = "E:/common_content/models/korean_public_payphone_01_4k/korean_public_payphone_01_4k.gltf";
        std::string train = "E:/common_content/models/steam_train/scene.gltf";
        std::string warrior = "E:/common_content/models/warrior_toy/scene.gltf";
        std::string trooper = "E:/costrmmon_content/models/StormTrooper/scene.gltf";
        std::string vase = "E:/common_content/models/antique_ceramic_vase_01_4k/antique_ceramic_vase_01_4k.gltf";
        std::string box = "E:/common_content/models/Box/glTF/Box.gltf";
        std::string bball = "C:/common_content/models/baseball_01_4k/baseball_01_4k.gltf";
        turtle = "C:/common_content/models/turtle/scene.gltf";
        std::string fish = "E:/common_content/models/BarramundiFish/glTF/BarramundiFish.gltf";
        std::string buggy = "E:/common_content/glTF-Sample-Models/2.0/Buggy/glTF/Buggy.gltf";
        std::string camera = "E:/common_content/glTF-Sample-Models/2.0/AntiqueCamera/glTF/AntiqueCamera.gltf";
        std::string ground = "E:/common_content/models/static_gound/static_ground.gltf";
        //  model.loadGLTF (ground);
      //   model.loadGLTF (helmet);
       model.loadGLTF (testModel);
     model.loadGLTF (box);
       // model.loadGLTF (cash);
        // model.loadGLTF (phone);
        //  model.loadGLTF (scifi);
        //  model.loadGLTF (warrior);
        //  model.loadGLTF (trooper);
        //   model.loadGLTF (helmet);
        std::vector<std::string> models;
        // models.push_back (testModel);
        // models.push_back (helmet);
        // models.push_back (cash);
        //  models.push_back (scifi);
        //  models.push_back (turtle);
        //  models.push_back (bust);
        //   models.push_back (phone);
        //  models.push_back (train);
        //  models.push_back (warrior);
        // models.push_back (trooper);
        // models.push_back (vase);
        // models.push_back (box);
        // model.onDrop (models);

        // Create a warm white mesh light with custom size and intensity
        warmLight = sabi::MeshOps::createLuminousRectangleNode (
            "WarmMeshLight",
            4.0f, 2.0f,                         // 4x2 units
            Eigen::Vector3f (1.0f, 0.9f, 0.8f), // Warm white color
            10.0f                               // High intensity
        );

        warmLight->setClientID (warmLight->getID());
        sabi::SpaceTime& st = warmLight->getSpaceTime();
        st.worldTransform.translation() = Eigen::Vector3f (0.0f, 1.0f, -4.0f);
       model.addNodeToRenderer (warmLight);

        {
            groundPlane = sabi::MeshOps::createGroundPlaneNode();
            groundPlane->setClientID (groundPlane->getID());
            sabi::SpaceTime& st = groundPlane->getSpaceTime();
            st.worldTransform.translation() = Eigen::Vector3f (0.0f, 0.0f, 0.0f);
         model.addNodeToRenderer (groundPlane);
        }
        /*  CgModelPtr c = sabi::MeshOps::createCube();

           cube = sabi::WorldItem::create();
          cube->setClientID (cube->getID());
           cube->setName ("Cube");
           cube->setModel (c);
           sabi::SpaceTime& st = cube->getSpaceTime();
           st.worldTransform.translation() = Eigen::Vector3f (0.0f, 1.0f, 0.0f);
           model.addNodeToRenderer (cube);*/
    }

    void update() override
    {
        /*  envRotation += 0.5;
          properties.renderProps->setValue (RenderKey::EnviroRotation, envRotation);

          if (envRotation >= 360) envRotation = 0.0;*/

        // Simple camera heading rotation - use DELTA rotation, not cumulative
        // float deltaRotation = 0.5f; // degrees per frame (same as env rotation)
        // float radians = deltaRotation * M_PI / 180.0f;
        // Eigen::Quaternionf rotation (Eigen::AngleAxisf (radians, Eigen::Vector3f::UnitY()));
        // view->getCamera()->rotateAroundTarget (rotation);
        // view->getCamera()->setDirty (true);

        // Update animation state in model from controller
        model.setAnimationEnabled (controller.isAnimationEnabled());

        model.onUpdate (lastInput);

        lastInput = InputEvent{};

        // get the render
        const OIIO::ImageBuf& img = view->getCamera()->getSensor()->getHDRImage();

        bool needsNewRenderTexture = (img.spec().width != lastImageWidth ||
                                      img.spec().height != lastImageHeight ||
                                      img.spec().nchannels != lastChannelCount);

        lastImageWidth = img.spec().width;
        lastImageHeight = img.spec().height;
        lastChannelCount = img.spec().nchannels;

        view->getCanvas()->updateRender (std::move (img), needsNewRenderTexture);

        if (socketServer)
        {
            socketServer->getMessenger().send (QMS::updateSocketServer());

            // After sending the update, check for new messages in the queue
            SocketServerImpl* const server = socketServer->getServer();
            if (!server) return;

            // Get the next message from the moody queue
            std::string cmd = server->getNextMoodyMessage();
            if (cmd.size())
            {
                LOG (DBUG) << "Processing command from queue: " << cmd;

                // Delegate command processing to the CommandProcessor
                std::string response = commandProcessor->processCommand (cmd);

                LOG (DBUG) << "Sending to client: " << response;

                // Send response back to client
                server->sendResponseToAllClients (response);
            }
        }
    }

    void onHDRLoaded (const std::filesystem::path& hdrPath)
    {
        view->setHDRFilename (hdrPath.string());
    }

    void onInputEvent (const mace::InputEvent& e) override
    {
        // no need to process moves is there?
        if (e.getType() == InputEvent::Type::Move) return;

        controller.onInputEvent (e, view->getCamera());

        // these events must be processed immediately by the back end for Picking
        mousePressRelease = e.getType() == InputEvent::Type::Press || e.getType() == InputEvent::Type::Release;

        if (mousePressRelease)
            model.onPriorityInput (e);
        lastInput = e;

        // lastMouseMode = mode;

        // if we're painting then send input to world
        if (e.getMouseMode() == MouseMode::Paint)
        {
            // model.onInputForPainting (e);
        }
    }

    void onCrash() override
    {
    }

 private:
    View* view = nullptr;
    Model model;
    Controller controller;
    PropertyService properties;
    double envRotation = 0.0;
    RenderableNode warmLight = nullptr;
    RenderableNode groundPlane = nullptr;
    RenderableNode cube = nullptr;
    // Socket server for remote connections
    std::unique_ptr<ActiveSocketServer> socketServer;
    MsgReceiver incoming;
    MessageService messengers;

    uint32_t lastImageWidth = 0;
    uint32_t lastImageHeight = 0;
    uint32_t lastChannelCount = 0;

    InputEvent lastInput;
    //  MouseMode lastMouseMode;
    bool mousePressRelease = false;

    // Command processor for handling remote commands
    std::unique_ptr<CommandProcessor> commandProcessor;
};

Jahley::App* Jahley::CreateApplication()
{
    // Add this before nanogui::init()
#ifdef _WIN32
    SetProcessDPIAware(); // Windows Vista/7/8
#endif

    nanogui::init();

    DesktopWindowSettings settings{};
    settings.name = APP_NAME;

    // handle DPI
#ifdef _WIN32
    HDC hdc = GetDC (NULL);
    int dpiX = GetDeviceCaps (hdc, LOGPIXELSX);
    int dpiY = GetDeviceCaps (hdc, LOGPIXELSY);
    float scaleX = dpiX / 96.0f; // 96 DPI is 100% scaling
    float scaleY = dpiY / 96.0f;
    ReleaseDC (NULL, hdc);

    settings.width = static_cast<int> (DEFAULT_DESKTOP_WINDOW_WIDTH / scaleX);
    settings.height = static_cast<int> (DEFAULT_DESKTOP_WINDOW_HEIGHT / scaleY);

#endif

    nanogui::ref<nanogui::Screen> screen = new View (settings);
    screen->set_visible (true);

    return new Application (settings, true);
}