
local ROOT = "../"

	language "C++"

	defines{
	 "_USE_MATH_DEFINES"
	}
	flags { "MultiProcessorCompile", "NoMinimalRebuild" }
	
	local CORE_DIR = ROOT .. "appCore/source/"
	local JAHLEY_DIR = ROOT .. "appCore/source/jahley/"
	local THIRD_PARTY_DIR = "../thirdparty/"
	local MODULE_DIR = "../modules/"
	
	includedirs
	{
		CORE_DIR,
		JAHLEY_DIR,
		MODULE_DIR,
		
		THIRD_PARTY_DIR,
		THIRD_PARTY_DIR .. "g3log/src",
		THIRD_PARTY_DIR .. "benchmark/include",
		THIRD_PARTY_DIR .. "json",
		THIRD_PARTY_DIR .. "stb_image",
		THIRD_PARTY_DIR .. "date/include/date",
		THIRD_PARTY_DIR .. "reproc++",
		THIRD_PARTY_DIR .. "cpptrace/include",
		THIRD_PARTY_DIR .. "libassert/include",
		THIRD_PARTY_DIR .. "binarytools/src",
		THIRD_PARTY_DIR .. "glfw/include",
		THIRD_PARTY_DIR .. "nanogui/include",
		THIRD_PARTY_DIR .. "nanogui/ext/glad/include",
		THIRD_PARTY_DIR .. "nanogui/ext/nanovg/src",
	}
	
	targetdir (ROOT .. "builds/bin/" .. outputdir .. "/%{prj.name}")
	objdir (ROOT .. "builds/bin-int/" .. outputdir .. "/%{prj.name}")
	
	filter { "system:windows"}
		disablewarnings { 
			"5030", "4244", "4267", "4667", "4018", "4101", "4305", "4316", "4146", "4996", "4554", "4060",
		} 
		linkoptions { "-IGNORE:4099" } -- can't find debug file in release folder
		characterset ("MBCS")
		buildoptions { "/Zm250", "/bigobj", "/Zc:__cplusplus"}
		
		defines 
		{ 
			"WIN32", "_WINDOWS", "NANOVG_GL3", "NANOGUI_USE_OPENGL",
			--https://github.com/KjellKod/g3log/issues/337
			"_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING","G3_DYNAMIC_LOGGING",
			"CHANGE_G3LOG_DEBUG_TO_DBUG","__TBB_NO_IMPLICIT_LINKAGE",
		}
		
	filter "configurations:Debug"
	
		postbuildcommands {
			
		}
		links 
		{ 
			"nanogui",
			"GLFW",
			"AppCore",
			"g3log",
			"benchmark",
			"opengl32",
			"stb_image", 
			"reproc++",
			"Ws2_32",
			"cpptrace",
			"libassert",
			"binarytools",
			
		}
			
			
		defines { "DEBUG", "USE_DEBUG_EXCEPTIONS" }
		symbols "On"
		libdirs { THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  MODULE_DIR .. "builds/bin/" .. outputdir .. "/**",
				  ROOT .. "builds/bin/" .. outputdir .. "/**",
				
		}
		
	filter "configurations:Release"
	postbuildcommands {
			
		}
		links 
		{ 
			"nanogui",
			"GLFW",
			"AppCore",
			"g3log",
			"benchmark",
			"opengl32",
			"stb_image", 
			"reproc++",
			"Ws2_32",
			"cpptrace",
			"libassert",
			"binarytools",
			
		}
		defines { "NDEBUG"}
		optimize "On"
		libdirs { THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  MODULE_DIR .. "builds/bin/" .. outputdir .. "/**",
				  ROOT .. "builds/bin/" .. outputdir .. "/**", 
		}
	
	  


	 		

