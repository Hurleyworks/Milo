project "g3log"
	if _ACTION == "vs2019" then
		cppdialect "C++17"
		location ("../builds/VisualStudio2019/projects")
	end
	if _ACTION == "vs2022" then
		cppdialect "C++20"
		location ("../builds/VisualStudio2022/projects")
    end
    kind "StaticLib"
    language "C++"
  
  
    flags { "MultiProcessorCompile" }
	targetdir ("../builds/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("../builds/bin-int/" .. outputdir .. "/%{prj.name}")
	
	includedirs
	{
        "src",
		"src/g3log",
    }

	files
	{
        "src/g3log/g3log.hpp",
        "src/filesink.cpp",
        "src/g2log.hpp",
        "src/g3log.cpp",
        "src/logcapture.cpp",
        "src/loglevels.cpp",
		"src/logmessage.cpp",
		"src/logworker.cpp",
		"src/time.cpp",
    }
    
	filter "system:windows"
	    staticruntime "On"
        systemversion "latest"
        disablewarnings { "4244", "5030" }
		characterset "MBCS"
		buildoptions { "/Zm250", "/bigobj","/Zc:__cplusplus",}
        files
        {
            "src/crashhandler_windows.cpp",
			"src/stacktrace_windows.cpp",
        }

		defines 
		{ 
			"_WIN32",
			"_WINDOWS",
            "_CRT_SECURE_NO_WARNINGS",
			--https://github.com/KjellKod/g3log/issues/337
			"_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING",
			"CHANGE_G3LOG_DEBUG_TO_DBUG","G3_DYNAMIC_LOGGING",
		}
	
