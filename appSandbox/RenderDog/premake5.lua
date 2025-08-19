local ROOT = "../../"

project  "RenderDog"
	if _ACTION == "vs2019" then
		cppdialect "C++17"
		location (ROOT .. "builds/VisualStudio2019/projects")
    end
	
	if _ACTION == "vs2022" then
		cppdialect "C++20"
		location (ROOT .. "builds/VisualStudio2022/projects")
    end
	
	kind "ConsoleApp"
	
    local FRAMEWORK_ROOT = ROOT .. "framework/"
	local QMS = FRAMEWORK_ROOT .. "qms_core/";
	local PROPS = FRAMEWORK_ROOT .. "properties_core/";
	local SERVER = FRAMEWORK_ROOT .. "server_core/";
	local GEOMETRY = FRAMEWORK_ROOT .. "geometry_core/";
	local IMAGING = FRAMEWORK_ROOT .. "image_core/";
	local NEWTON = FRAMEWORK_ROOT .. "physics_core/";
	local RENDER = FRAMEWORK_ROOT .. "milo_core/";
	local WORLD = FRAMEWORK_ROOT .. "world_core/";


	local SOURCE_DIR = "source/*"
    files
    { 
      SOURCE_DIR .. "**.h", 
      SOURCE_DIR .. "**.hpp", 
      SOURCE_DIR .. "**.c",
      SOURCE_DIR .. "**.cpp",
	  
	  QMS .. "**.h", 
	  QMS .. "**.cpp",
	  PROPS .. "**.h", 
	  PROPS .. "**.cpp",
	  SERVER .. "**.h", 
	  SERVER .. "**.cpp",
	  GEOMETRY .. "**.h", 
	  GEOMETRY .. "**.cpp",
	  IMAGING .. "**.h", 
	  IMAGING .. "**.cpp",
	  NEWTON .. "**.h", 
	  NEWTON .. "**.cpp",
	  RENDER .. "**.h", 
	  RENDER .. "**.cpp",
	  RENDER .. "**.cuh", 
	  RENDER .. "**.cu",
	  WORLD .. "**.h", 
	  WORLD .. "**.cpp",
    }
	
	includedirs
	{
	    FRAMEWORK_ROOT
	}
	
	filter "system:windows"
		staticruntime "On"
		systemversion "latest"
		defines {"_CRT_SECURE_NO_WARNINGS", 
		"	CPPTRACE_STATIC_DEFINE", "NOMINMAX",
			"CPPTRACE_GET_SYMBOLS_WITH_DBGHELP",
			"CPPTRACE_UNWIND_WITH_DBGHELP",
			"CPPTRACE_DEMANGLE_WITH_WINAPI",
			"LIBASSERT_LOWERCASE",
			"LIBASSERT_SAFE_COMPARISONS",
			"LIBASSERT_STATIC_DEFINE"
		}
		
		disablewarnings { "5030" , "4305", "4316", "4267"}
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
		
	filter {} -- clear filter!
	filter { "files:../../framework/**/excludeFromBuild/**.cpp"}
	flags {"ExcludeFromBuild"}
	filter {} -- clear filter!
		
-- add settings common to all project
dofile("../../buildTools/render_common.lua")


