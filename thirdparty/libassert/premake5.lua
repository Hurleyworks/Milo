  
   
	
project "libassert"
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
		"include/",
		"../cpptrace/include",
	}
	files
	{
		"include/**.hpp", 
		"include/**.h", 
		
		"src/**.chpp", 
		"src/**.cpp", 
		"src/**.c", 
    }
	
	
	filter "configurations:Release"
        optimize "On"
    
	filter "system:windows"
        staticruntime "On"
		characterset ("MBCS")
		buildoptions { "/Zm250"}
        disablewarnings { "4244", "4101", "4267", "4018" }
        files
        {
			
        }

		defines 
		{ 
            "_CRT_SECURE_NO_WARNINGS", "CPPTRACE_STATIC_DEFINE", "NOMINMAX",
			"CPPTRACE_GET_SYMBOLS_WITH_DBGHELP",
			"CPPTRACE_UNWIND_WITH_DBGHELP",
			"CPPTRACE_DEMANGLE_WITH_WINAPI",
			--"LIBASSERT_USE_MAGIC_ENUM",
			"LIBASSERT_LOWERCASE",
			"LIBASSERT_SAFE_COMPARISONS",
			"LIBASSERT_STATIC_DEFINE"

		}
		filter { "system:windows", "configurations:Release"}
			buildoptions "/MT"     

		filter { "system:windows", "configurations:Debug"}
			buildoptions "/MTd"  
			
