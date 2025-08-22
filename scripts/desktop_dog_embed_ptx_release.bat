@echo off
echo Running new embed_ptx.py...
"C:\Users\Steve\AppData\Local\Programs\Python\Python311\python.exe" "E:\1Milo\Milo\scripts\embed_ptx.py" ^
"E:\1Milo\Milo\resources\RenderDog\ptx" "E:\1Milo\Milo\framework\claude_core\generated\embedded_ptx.h" "Release"
echo Exit code: %ERRORLEVEL%
echo Checking if script completed...
if exist "E:\1Milo\Milo\framework\claude_core\generated\embedded_ptx.h" (
  echo SUCCESS: Output file was created
) else (
  echo ERROR: Output file was not created
)
pause