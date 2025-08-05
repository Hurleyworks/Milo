project "GLFW"
	if _ACTION == "vs2019" then
		cppdialect "C++17"
		location ("../builds/VisualStudio2019/projects")
	end
	if _ACTION == "vs2022" then
		cppdialect "C++20"
		location ("../builds/VisualStudio2022/projects")
    end
    kind "StaticLib"
    language "C"
	
    flags { "MultiProcessorCompile" }
	targetdir ("../builds/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("../builds/bin-int/" .. outputdir .. "/%{prj.name}")

	files
	{
        "include/GLFW/glfw3.h",
        "include/GLFW/glfw3native.h",
        "src/glfw_config.h",
        "src/context.c",
        "src/init.c",
        "src/input.c",
        "src/monitor.c",
        "src/vulkan.c",
        "src/window.c"
    }
    
	filter "system:windows"
        systemversion "latest"
        staticruntime "On"
        buildoptions { "/Zm250", "/bigobj","/Zc:__cplusplus",}
        files
        {
            "src/win32_init.c",
            "src/win32_joystick.c",
            "src/win32_monitor.c",
            "src/win32_time.c",
            "src/win32_thread.c",
            "src/win32_window.c",
            "src/wgl_context.c",
            "src/egl_context.c",
            "src/osmesa_context.c"
        }

		defines 
		{ 
            "_GLFW_WIN32",
            "_CRT_SECURE_NO_WARNINGS"
		}
    filter { "system:windows", "configurations:Release" }
        buildoptions "/MT"
