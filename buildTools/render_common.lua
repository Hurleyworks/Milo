
local ROOT = "../"

	language "C++"

	defines{
	 "_USE_MATH_DEFINES"
	}
	flags { "MultiProcessorCompile", "NoMinimalRebuild" }
	
	local CORE_DIR = ROOT .. "appCore/source/"
	local JAHLEY_DIR = ROOT .. "appCore/source/jahley/"
	local THIRD_PARTY_DIR = "../thirdparty/"
	local VCPKG_DIR = ROOT .. "thirdparty/vcpkg/installed/x64-windows-static/"
	local MODULE_DIR = "../modules/"
	
	
	local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/extras/cupti/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64"
	
	local OPTIX_ROOT = "C:/ProgramData/NVIDIA Corporation"
	local OPTIX8_INCLUDE_DIR = OPTIX_ROOT .. "/OptiX SDK 9.0.0/include"
	
	includedirs
	{
		CORE_DIR,
		JAHLEY_DIR,
		MODULE_DIR,
		
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		OPTIX8_INCLUDE_DIR,
		
		
		-- for OIIO
		VCPKG_DIR,
		VCPKG_DIR .. "include",
		
		
		THIRD_PARTY_DIR,
		THIRD_PARTY_DIR .. "g3log/src",
		THIRD_PARTY_DIR .. "json",
		THIRD_PARTY_DIR .. "optiXUtil/src",
		THIRD_PARTY_DIR .. "stb_image",
		THIRD_PARTY_DIR .. "newtondynamics/sdk/**",
		THIRD_PARTY_DIR .. "newtondynamics/**",
		THIRD_PARTY_DIR .. "date/include/date",
		THIRD_PARTY_DIR .. "reproc++",
		THIRD_PARTY_DIR .. "glfw/include",
		THIRD_PARTY_DIR .. "binarytools/src",
		THIRD_PARTY_DIR .. "cpptrace/include",
		THIRD_PARTY_DIR .. "libassert/include",
		THIRD_PARTY_DIR .. "nanogui/include",
		THIRD_PARTY_DIR .. "nanogui/ext/glad/include",
		THIRD_PARTY_DIR .. "nanogui/ext/nanovg/src",
		THIRD_PARTY_DIR .. "fastgltf/include",
		THIRD_PARTY_DIR .. "earcut/include/mapbox",
	}
	
	targetdir (ROOT .. "builds/bin/" .. outputdir .. "/%{prj.name}")
	objdir (ROOT .. "builds/bin-int/" .. outputdir .. "/%{prj.name}")
	
	filter { "system:windows"}
		disablewarnings { 
			"5030", "4244", "4267", "4667", "4018", "4101", "4305", "4316", "4146", "4996", "4554", "4060", "4200",
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
			"nanogui",
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
			"fastgltf",
			"optiXUtil",
			"newtondynamics",
			"reproc++",
			
			--cuda
			"cudart_static",
			"cuda",
			"nvrtc",
			"cublas",
			"curand",
			"cusolver",
			"cudart",
			
			--cubd
			"libcubd_staticd",
			
			--oiio
			"boost_thread-vc144-mt-gd-x64-1_85",
			"boost_filesystem-vc144-mt-gd-x64-1_85",
			"boost_system-vc144-mt-gd-x64-1_85",
			"Iex-3_2_d",
			"IlmThread-3_2_d",
			"Imath-3_1_d",
			"jpeg",
			"turbojpeg",
			"tiffd",
			"zlibd",
			"lzma",
			"deflatestatic",
			"libpng16d",
			"OpenEXR-3_2_d",
			"OpenEXRCore-3_2_d",
			"OpenEXRUtil-3_2_d",
			"OpenImageIO_d",
			"OpenImageIO_Util_d",
			
		}
			
			
		defines { "DEBUG", "USE_DEBUG_EXCEPTIONS" }
		symbols "On"
		libdirs { THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  VCPKG_DIR .. "debug/lib",
				  ROOT .. "builds/bin/" .. outputdir .. "/**",
				  CUDA_LIB_DIR,
				
		}
		
	filter "configurations:Release"
	postbuildcommands {
			
		}
		links 
		{ 
			"nanogui",
			"GLFW",
			"nanogui",
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
			"fastgltf",
			"optiXUtil",
			"newtondynamics",
			"reproc++",
			
			--cuda
			"cudart_static",
			"cuda",
			"nvrtc",
			"cublas",
			"curand",
			"cusolver",
			"cudart",
			
			--cubd
			"libcubd_static",
			
			  --for 0ii0
			"boost_thread-vc144-mt-x64-1_85",
			"boost_filesystem-vc144-mt-x64-1_85",
			"boost_system-vc144-mt-x64-1_85",
			"Iex-3_2",
			"IlmThread-3_2",
			"Imath-3_1",
			"jpeg",
			"turbojpeg",
			"tiff",
			"zlib",
			"lzma",
			"deflatestatic",
			"libpng16",
			"OpenEXR-3_2",
			"OpenEXRCore-3_2",
			"OpenEXRUtil-3_2",
			"OpenImageIO",
			"OpenImageIO_Util",
		}
		defines { "NDEBUG"}
		optimize "On"
		libdirs { THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  VCPKG_DIR .. "lib", 
				  ROOT .. "builds/bin/" .. outputdir .. "/**", 
				   CUDA_LIB_DIR,
		}
	
	  


	 		

