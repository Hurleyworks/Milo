
local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/extras/cupti/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/lib/x64"
	
project "newtondynamics"
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
	
	defines {"_NEWTON_STATIC_LIB", "_CUSTOM_JOINTS_STATIC_LIB", "_DVEHICLE_STATIC_LIB","_CNEWTON_STATIC_LIB"}

	includedirs
	{
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		"openFBX/*",
		"tinyxml",
        "sdk/*",
		"sdk/dModel/*",
		"sdk/dNewton/**",
		--"source/**",
    }

	files
	{
		"sdk/**.h", 
		"sdk/**.cpp", 
		"sdk/**.c", 
		"sdk/**.cc", 
    }
    
	filter "system:windows"
        staticruntime "On"
        systemversion "latest"
       
		defines 
		{ 
            "_WIN_64_VER","_WINDOWS","WIN32",
			"_D_SINGLE_LIBRARY",
			"_CRT_SECURE_NO_WARNINGS"
		}

    filter "configurations:Debug"
        symbols "On"
                
    filter "configurations:Release"
        optimize "On"






