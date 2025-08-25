
-- https://github.com/JohannesMP/Premake-for-Beginners

workspace "UnitTests"
	architecture "x64"
	location ("builds")
	
if _ACTION == "vs2022" then
   location ("builds/VisualStudio2022")
end
if _ACTION == "vs2019" then
   location ("builds/VisualStudio2019")
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

	
	include "tests/HelloTest"
	include "tests/CgModel2Shocker"
	include "tests/AreaLightTest"