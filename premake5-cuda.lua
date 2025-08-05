require("buildTools/cuda-exported-variables")

if os.target() == "windows" then
    dofile("buildTools/premake5-cuda-vs.lua")
end
