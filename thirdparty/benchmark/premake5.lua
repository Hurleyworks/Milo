
project "benchmark"

if _ACTION == "vs2019" then
	location ("../builds/VisualStudio2019/projects")
	 cppdialect "C++17"
end
if _ACTION == "vs2022" then
	location ("../builds/VisualStudio2022/projects")
	 cppdialect "C++20"
end
    kind "StaticLib"
    language "C++"
   
    flags { "MultiProcessorCompile" }
	defines {"HAVE_STD_REGEX", "HAVE_STEADY_CLOCK","BENCHMARK_STATIC_DEFINE"}
	targetdir ("../builds/bin/" .. outputdir .. "/%{prj.name}")
    objdir ("../builds/bin-int/" .. outputdir .. "/%{prj.name}")
	
	includedirs
	{
		
		"include",
        
    }

	files
	{
		"src/**.h", 
		"src/**.cc", 
    }
	
	
	
	filter "system:windows"
        staticruntime "On"
        systemversion "latest"
        disablewarnings { "4244", "4267", "4996", "4305", "4018", "4334", "4312", "4311", "4251", "4275"}
		defines { "_CRT_SECURE_NO_WARNINGS", "BENCHMARK_OS_WINDOWS"}
		characterset "MBCS"
		buildoptions { "/Zm250", "/bigobj","/Zc:__cplusplus",}

    filter "configurations:Debug"
        symbols "On"
                
    filter "configurations:Release"
        optimize "On"
	