  
    local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/extras/cupti/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64"
	local OPTIX_ROOT = "C:/ProgramData/NVIDIA Corporation"
	local OPTIX8_INCLUDE_DIR = OPTIX_ROOT .. "/OptiX SDK 9.0.0/include"
	
project "optiXUtil"
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
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		OPTIX8_INCLUDE_DIR,
	}
	files
	{
		"src/**.h", 
		"src/**.cpp", 
    }
	
	
	filter "configurations:Release"
        optimize "On"
    
	filter "system:windows"
        staticruntime "On"
		characterset ("MBCS")
		buildoptions { "/Zm250", "/Zc:__cplusplus"}
        disablewarnings { "4244", "4101", "4267", "4018" }
        files
        {
			
        }

		defines 
		{ 
            "_CRT_SECURE_NO_WARNINGS",
		}
		filter { "system:windows", "configurations:Release"}
			buildoptions "/MT"     

		filter { "system:windows", "configurations:Debug"}
			buildoptions "/MTd"  
			
