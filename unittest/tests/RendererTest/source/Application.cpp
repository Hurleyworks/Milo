// RendererTest - Tests for Renderer and RenderContext lifetime management
// Verifies proper initialization, cleanup, and resource management

#include "dog_core.h"
#include "dog_core/excludeFromBuild/Renderer.h"
#include "dog_core/excludeFromBuild/RenderContext.h"
#include "dog_core/excludeFromBuild/handlers/Handlers.h"

#include <doctest.h>
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>

// Test fixture for Renderer tests
class RendererTestFixture 
{
public:
    std::unique_ptr<g3::LogWorker> logworker;
    
    RendererTestFixture() 
    {
        // Initialize logging for tests
        logworker = g3::LogWorker::createLogWorker();
        auto handle = logworker->addDefaultLogger("RendererTest", "./logs/");
        g3::initializeLogging(logworker.get());
    }
    
    ~RendererTestFixture() 
    {
        // Cleanup logging
        logworker.reset();
    }
};

TEST_CASE("Renderer lifetime management") 
{
    RendererTestFixture fixture;
    
    SUBCASE("Basic initialization and shutdown") 
    {
        LOG(INFO) << "Testing basic Renderer initialization and shutdown";
        
        Renderer renderer;
        CHECK_FALSE(renderer.isInitialized());
        
        // Initialize renderer
        bool result = renderer.initialize(0, 1024, 768);
        CHECK(result);
        CHECK(renderer.isInitialized());
        CHECK(renderer.getWidth() == 1024);
        CHECK(renderer.getHeight() == 768);
        CHECK(renderer.getRenderContext() != nullptr);
        
        // Verify render context is properly initialized
        auto context = renderer.getRenderContext();
        CHECK(context->isInitialized());
        CHECK(context->getRenderWidth() == 1024);
        CHECK(context->getRenderHeight() == 768);
        
        // Shutdown
        renderer.shutdown();
        CHECK_FALSE(renderer.isInitialized());
        CHECK(renderer.getRenderContext() == nullptr);
        
        LOG(INFO) << "Basic initialization test passed";
    }
    
    SUBCASE("Double initialization protection") 
    {
        LOG(INFO) << "Testing double initialization protection";
        
        Renderer renderer;
        
        // First initialization
        bool result1 = renderer.initialize(0, 800, 600);
        CHECK(result1);
        CHECK(renderer.isInitialized());
        
        // Second initialization should return true but not reinitialize
        bool result2 = renderer.initialize(0, 1920, 1080);
        CHECK(result2);
        CHECK(renderer.isInitialized());
        CHECK(renderer.getWidth() == 800);  // Should keep original size
        CHECK(renderer.getHeight() == 600);
        
        LOG(INFO) << "Double initialization protection test passed";
    }
    
    SUBCASE("Destructor cleanup") 
    {
        LOG(INFO) << "Testing destructor cleanup";
        
        {
            Renderer renderer;
            renderer.initialize(0, 640, 480);
            CHECK(renderer.isInitialized());
            // Destructor should be called here
        }
        
        // Create another renderer to verify GPU resources were properly released
        Renderer renderer2;
        bool result = renderer2.initialize(0, 640, 480);
        CHECK(result);
        CHECK(renderer2.isInitialized());
        
        LOG(INFO) << "Destructor cleanup test passed";
    }
    
    SUBCASE("Resize functionality") 
    {
        LOG(INFO) << "Testing resize functionality";
        
        Renderer renderer;
        renderer.initialize(0, 800, 600);
        CHECK(renderer.getWidth() == 800);
        CHECK(renderer.getHeight() == 600);
        
        // Resize
        renderer.resize(1920, 1080);
        CHECK(renderer.getWidth() == 1920);
        CHECK(renderer.getHeight() == 1080);
        
        // Verify context was updated
        auto context = renderer.getRenderContext();
        CHECK(context->getRenderWidth() == 1920);
        CHECK(context->getRenderHeight() == 1080);
        
        // Verify screen buffers were resized
        auto handlers = context->getHandlers();
        if (handlers && handlers->screenBuffer) 
        {
            CHECK(handlers->screenBuffer->getWidth() == 1920);
            CHECK(handlers->screenBuffer->getHeight() == 1080);
        }
        
        // Test no-op resize
        renderer.resize(1920, 1080);
        CHECK(renderer.getWidth() == 1920);
        CHECK(renderer.getHeight() == 1080);
        
        LOG(INFO) << "Resize functionality test passed";
    }
    
    SUBCASE("Frame operations") 
    {
        LOG(INFO) << "Testing frame operations";
        
        Renderer renderer;
        
        // Operations should fail when not initialized
        renderer.beginFrame();  // Should log warning but not crash
        renderer.endFrame();    // Should log warning but not crash
        renderer.present();     // Should log warning but not crash
        
        // Initialize and test frame operations
        renderer.initialize(0, 800, 600);
        
        // Simulate rendering a few frames
        for (int i = 0; i < 3; ++i) 
        {
            renderer.beginFrame();
            renderer.endFrame();
            renderer.present();
        }
        
        LOG(INFO) << "Frame operations test passed";
    }
}

TEST_CASE("RenderContext handler management") 
{
    RendererTestFixture fixture;
    
    SUBCASE("Handler initialization") 
    {
        LOG(INFO) << "Testing handler initialization";
        
        auto context = RenderContext::create();
        CHECK(context != nullptr);
        
        bool result = context->initialize(0);
        CHECK(result);
        CHECK(context->isInitialized());
        
        // Verify handlers were created
        auto handlers = context->getHandlers();
        CHECK(handlers != nullptr);
        
        // Verify screen buffer handler was initialized
        CHECK(handlers->screenBuffer != nullptr);
        CHECK(handlers->screenBuffer->isInitialized());
        
        // Check default dimensions
        CHECK(handlers->screenBuffer->getWidth() == 1920);
        CHECK(handlers->screenBuffer->getHeight() == 1080);
        
        context->cleanup();
        CHECK_FALSE(context->isInitialized());
        
        LOG(INFO) << "Handler initialization test passed";
    }
    
    SUBCASE("Handler cleanup order") 
    {
        LOG(INFO) << "Testing handler cleanup order";
        
        {
            auto context = RenderContext::create();
            context->initialize(0);
            
            // Get handlers pointer before cleanup
            auto handlers = context->getHandlers();
            CHECK(handlers != nullptr);
            CHECK(handlers->screenBuffer != nullptr);
            
            // Context cleanup should properly clean up handlers
            context->cleanup();
            
            // After cleanup, handlers should be null
            CHECK(context->getHandlers() == nullptr);
        }
        
        LOG(INFO) << "Handler cleanup order test passed";
    }
}

TEST_CASE("Memory leak detection") 
{
    RendererTestFixture fixture;
    
    SUBCASE("Multiple init/shutdown cycles") 
    {
        LOG(INFO) << "Testing multiple init/shutdown cycles for memory leaks";
        
        Renderer renderer;
        
        // Perform multiple init/shutdown cycles
        for (int i = 0; i < 3; ++i) 
        {
            LOG(INFO) << "Cycle " << (i + 1);
            
            bool result = renderer.initialize(0, 800 + i * 100, 600 + i * 100);
            CHECK(result);
            CHECK(renderer.isInitialized());
            
            // Do some operations
            renderer.beginFrame();
            renderer.endFrame();
            renderer.present();
            
            renderer.shutdown();
            CHECK_FALSE(renderer.isInitialized());
        }
        
        LOG(INFO) << "Multiple init/shutdown cycles test passed";
    }
}

int main(int argc, char** argv) 
{
    // Initialize doctest
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    
    // Run tests
    int res = context.run();
    
    return res;
}