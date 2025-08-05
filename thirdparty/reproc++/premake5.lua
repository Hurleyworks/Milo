project "reproc++"
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
	   "../",
	    "./",
      "reproc/include",
	   "reproc++",
    }

	filter "system:windows"
	    staticruntime "On"
        systemversion "latest"
        disablewarnings { "4244", "5030" }
		characterset "MBCS"
		
        files { "reproc/src/*.c", "src/*.cpp",  }
		removefiles { "reproc/src/**.posix.c" }

		defines 
		{ 
			"WIN32",
			"_WINDOWS",
			"REPROC_MULTITHREADED",
			"WIN32_LEAN_AND_MEAN",	
			"_CRT_SECURE_NO_WARNINGS"
		}
	
