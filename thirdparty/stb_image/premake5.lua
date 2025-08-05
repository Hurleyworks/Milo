
local ROOT = "../"

project "stb_image"
	if _ACTION == "vs2019" then
		cppdialect "C++17"
		location (ROOT .. "builds/VisualStudio2019/projects")
    end
	
	if _ACTION == "vs2022" then
		cppdialect "C++20"
		location (ROOT .. "builds/VisualStudio2022/projects")
    end
	
    kind "StaticLib"
    language "C++"
    
    flags { "MultiProcessorCompile" }
	
	targetdir (ROOT .. "builds/bin/" .. outputdir .. "/%{prj.name}")
	objdir (ROOT .. "builds/bin-int/" .. outputdir .. "/%{prj.name}")
	
	includedirs
	{
        "./"
    }

	files
	{
		"stbi_wrapper.cpp", 
		
    }
	
	 filter "configurations:Release"
        optimize "On"
    
	filter "system:windows"
        staticruntime "On"
        disablewarnings { "4244" }
		buildoptions { "/Zm250", "/Zc:__cplusplus"}
        files
        {
			
        }

		defines 
		{ 
            "_CRT_SECURE_NO_WARNINGS"
		}
		filter { "system:windows", "configurations:Release"}
			buildoptions "/MT"     

		filter { "system:windows", "configurations:Debug"}
			buildoptions "/MTd"  
			
