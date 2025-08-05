
-- https://github.com/JohannesMP/Premake-for-Beginners

-- Include the premake5 CUDA module
require('premake5-cuda')

workspace "Apps"
	architecture "x64"
	location ("builds")

if _ACTION == "vs2019" then
   location ("builds/VisualStudio2019")
end
if _ACTION == "vs2022" then
   location ("builds/VisualStudio2022")
end

	configurations 
	{ 
		"Debug", 
        "Release",
    }
	vectorextensions "AVX2"
	filter "configurations:Debug"    defines { "DEBUG" }  symbols  "On"
    filter "configurations:Release"  defines { "NDEBUG" } optimize "On"
    
	outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
	
local ROOT = "../../"

-- the core
project "AppCore"
     kind "StaticLib"
    language "C++"
	defines { "_ENABLE_EXTENDED_ALIGNED_STORAGE", "WIN32","G3_DYNAMIC_LOGGING",
			"OIIO_STATIC_DEFINE", "CHANGE_G3LOG_DEBUG_TO_DBUG"}
	flags { "MultiProcessorCompile", "NoMinimalRebuild" }
	
	if _ACTION == "vs2019" then
		cppdialect "C++17"
		location "builds/VisualStudio2019/projects"
	end
	if _ACTION == "vs2022" then
		cppdialect "C++20"
		location ("builds/VisualStudio2022/projects")
	end
	
	pchheader "berserkpch.h"
    pchsource "appCore/source/berserkpch.cpp"
	
	targetdir ("builds/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("builds/bin-int/" .. outputdir .. "/%{prj.name}")

	local SOURCE_DIR = "appCore/source/"
	local BERSERKO_DIR = "appCore/source/jahley/"
	local MODULE_ROOT = "modules/"

    files
    { 
      SOURCE_DIR .. "**.h", 
      SOURCE_DIR .. "**.hpp", 
      SOURCE_DIR .. "**.c",
      SOURCE_DIR .. "**.cpp",
      SOURCE_DIR .. "**.tpp",
	  
	   MODULE_ROOT .. "**.h",  
      MODULE_ROOT .. "**.cpp",
	  MODULE_ROOT .. "**.cu",
   }
	
	local THIRD_PARTY_DIR = "thirdparty/"
	local VCPKG_DIR = "thirdparty/vcpkg/installed/x64-windows-static/"
	
	includedirs
	{
		SOURCE_DIR,
		BERSERKO_DIR,
		MODULE_ROOT,
		VCPKG_DIR,
		
		THIRD_PARTY_DIR,
		THIRD_PARTY_DIR .. "g3log/src/",
		THIRD_PARTY_DIR .. "binarytools/src/",
		THIRD_PARTY_DIR .. "cpptrace/include/",
		THIRD_PARTY_DIR .. "libassert/include/",
		THIRD_PARTY_DIR .. "fastgltf/include/",
		THIRD_PARTY_DIR .. "glfw/include",
		THIRD_PARTY_DIR .. "nanogui/include",
		THIRD_PARTY_DIR .. "nanogui/ext/glad/include",
		THIRD_PARTY_DIR .. "nanogui/ext/nanovg/src",
		VCPKG_DIR .. "include",
	}
	
	filter {} -- clear filter!
	
	 -- must go after files and includedirs
    filter { "files:modules/**/excludeFromBuild/**.cpp"}
	flags {"ExcludeFromBuild"}
	
	
	filter "system:windows"
        staticruntime "On"
        systemversion "latest"
		characterset ("MBCS")
        disablewarnings { 
			"5030", "4244", "4267", "4667", "4018", "4101", "4305", "4316", "4146", "4996",
		} 
		buildoptions { "/Zm250", "/bigobj","/Zc:__cplusplus",}
		
		defines 
		{ 
			"_CRT_SECURE_NO_WARNINGS",
			--https://github.com/KjellKod/g3log/issues/337
			"_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING",
			"G3_DYNAMIC_LOGGING",
			"_USE_MATH_DEFINES",
			 "CPPTRACE_STATIC_DEFINE", "NOMINMAX",
			"CPPTRACE_GET_SYMBOLS_WITH_DBGHELP",
			"CPPTRACE_UNWIND_WITH_DBGHELP",
			"CPPTRACE_DEMANGLE_WITH_WINAPI",
			"LIBASSERT_LOWERCASE",
			"LIBASSERT_SAFE_COMPARISONS",
			"LIBASSERT_STATIC_DEFINE",
			
		}
		
		 -- setting up visual studio filters (basically virtual folders).
		vpaths 
		{
		  ["Header Files/*"] = { 
			SOURCE_DIR .. "**.h", 
			SOURCE_DIR .. "**.hxx", 
			SOURCE_DIR .. "**.hpp",
		  },
		  ["Source Files/*"] = { 
			SOURCE_DIR .. "**.c", 
			SOURCE_DIR .. "**.cxx", 
			SOURCE_DIR .. "**.cpp",
		  },
		}
	
	filter "configurations:Debug"
        defines {"USE_DEBUG_EXCEPTIONS"}
    
	
	-- add projects here
	include "appSandbox/HelloWorld"
	include "appSandbox/RenderDog"
	
