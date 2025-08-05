import sys
import os
import traceback

def main():
    try:
        print(f"Script started")
        print(f"Arguments: {sys.argv}")
        if len(sys.argv) < 3:
            print("Usage: python embed_ptx.py <input_dir> <output_file> [build_mode]")
            sys.exit(1)
        
        input_dir = sys.argv[1]
        output_file = sys.argv[2]
        build_mode = sys.argv[3] if len(sys.argv) > 3 else "Release"
        
        print(f"Input directory: {input_dir}")
        print(f"Output file: {output_file}")
        print(f"Build mode: {build_mode}")
        
        base_dir = os.path.join(input_dir, build_mode)
        print(f"Looking for architecture folders in: {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"Error: Input directory does not exist: {base_dir}")
            sys.exit(1)
        
        arch_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("sm_")]
        base_ptx_files = [f for f in os.listdir(base_dir) if f.endswith(('.ptx', '.optixir'))]
        
        print(f"Found {len(arch_dirs)} architecture directories:")
        for d in arch_dirs:
            print(f"  {d}")
        
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            print(f"Output directory created/verified: {os.path.dirname(output_file)}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            raise
        
        kernels_by_arch = {}
        
        for arch in arch_dirs:
            arch_path = os.path.join(base_dir, arch)
            arch_files = [f for f in os.listdir(arch_path) if f.endswith(('.ptx', '.optixir'))]
            kernels_by_arch[arch] = []
            for filename in arch_files:
                kernel_name = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                kernels_by_arch[arch].append((kernel_name, ext))
        
        with open(output_file, 'w') as out:
            print(f"Writing to output file: {output_file}")
            out.write("#pragma once\n\n")
            
            if arch_dirs:
                out.write("#define EMBEDDED_PTX_AVAILABLE\n\n")
            
            out.write("#include <map>\n")
            out.write("#include <string>\n")
            out.write("#include <vector>\n\n")
            out.write("namespace embedded {\n\n")
            out.write("// Structure to hold PTX/OptiXIR data\n")
            out.write("struct PTXData {\n")
            out.write("    const unsigned char* data;\n")
            out.write("    size_t size;\n")
            out.write("    const char* format;  // \"ptx\" or \"optixir\"\n")
            out.write("};\n\n")
            
            # Process each architecture directory
            for arch in sorted(arch_dirs):
                arch_path = os.path.join(base_dir, arch)
                arch_files = [f for f in os.listdir(arch_path) if f.endswith(('.ptx', '.optixir'))]
                
                if not arch_files:
                    continue
                
                # Extract compute capability as integer (e.g., sm_86 -> 86)
                compute_capability = int(arch.split('_')[1])
                
                # Create namespace for this architecture
                out.write(f"// Architecture {arch} PTX/OptiXIR files\n")
                out.write(f"namespace {arch} {{\n\n")
                
                # Process each file in this architecture directory
                for filename in sorted(arch_files):
                    full_path = os.path.join(arch_path, filename)
                    with open(full_path, 'rb') as f:
                        content = f.read()
                    
                    # Add null terminator for PTX files if not already present
                    ext = os.path.splitext(filename)[1].lower()
                    if ext == '.ptx' and (len(content) == 0 or content[-1] != 0):
                        print(f"  Adding null terminator to PTX file: {filename} ({arch})")
                        content = content + b'\0'  # Add null byte for PTX files
                    
                    # Create safe variable names
                    base_name = os.path.splitext(filename)[0]
                    ext = os.path.splitext(filename)[1][1:]  # Remove the dot
                    var_name = f"{base_name}_data"
                    size_name = f"{base_name}_data_size"
                    
                    # Write the data as a byte array
                    bytes_str = ','.join([f"0x{b:02x}" for b in content])
                    out.write(f"// Generated from {filename} ({arch})\n")
                    out.write(f"static const unsigned char {var_name}[] = {{\n")
                    out.write(f"    {bytes_str}\n")
                    out.write("};\n")
                    out.write(f"static const size_t {size_name} = sizeof({var_name});\n\n")
                
                out.write(f"}} // namespace {arch}\n\n")
            
            # Create a mapping of which kernels are available for which architectures
            all_kernels = set()
            for arch, kernels in kernels_by_arch.items():
                for kernel_name, ext in kernels:
                    all_kernels.add(kernel_name)
            
            # Create helper functions for runtime selection
            out.write("// Helper function to get the best matching PTX/OptiXIR data for a compute capability\n")
            out.write("inline std::pair<const unsigned char*, size_t> getBestMatchForCompute(\n")
            out.write("    const std::string& kernelName, int computeCapability) {\n")
            
            # Create the selection logic
            for kernel_name in sorted(all_kernels):
                # Find which architectures have this kernel
                available_arches = []
                for arch in arch_dirs:
                    for k_name, ext in kernels_by_arch[arch]:
                        if k_name == kernel_name:
                            available_arches.append((arch, int(arch.split('_')[1]), ext))
                
                if not available_arches:
                    continue
                
                # Sort by compute capability, highest first
                available_arches.sort(key=lambda x: x[1], reverse=True)
                
                out.write(f"\n    // Handler for {kernel_name}\n")
                out.write(f"    if (kernelName == \"{kernel_name}\") {{\n")
                
                # Generate selection code for each available architecture
                for arch, compute_cap, ext in available_arches:
                    out.write(f"        if (computeCapability >= {compute_cap}) {{\n")
                    out.write(f"            return {{{arch}::{kernel_name}_data, ")
                    out.write(f"{arch}::{kernel_name}_data_size}};\n")
                    out.write(f"        }}\n")
                
                # If compute capability is lower than all available versions,
                # use the lowest available version as fallback
                lowest_arch = min(available_arches, key=lambda x: x[1])
                out.write(f"        // Fallback to lowest available architecture\n")
                out.write(f"        return {{{lowest_arch[0]}::{kernel_name}_data, ")
                out.write(f"{lowest_arch[0]}::{kernel_name}_data_size}};\n")
                out.write(f"    }}\n")
            
            # End of getBestMatchForCompute function
            out.write("\n    throw std::runtime_error(\"Unknown kernel name: \" + kernelName);\n")
            out.write("}\n\n")
            
            # Add function to check file format
            out.write("// Helper function to get the format (ptx or optixir) for a kernel and compute capability\n")
            out.write("inline std::string getFormatForCompute(const std::string& kernelName, int computeCapability) {\n")
            
            # Determine format based on file extension
            for kernel_name in sorted(all_kernels):
                formats_by_arch = {}
                for arch in arch_dirs:
                    for k_name, ext in kernels_by_arch[arch]:
                        if k_name == kernel_name:
                            ext_format = "ptx" if ext.lower() == ".ptx" else "optixir"
                            if arch not in formats_by_arch:
                                formats_by_arch[arch] = ext_format
                
                if formats_by_arch:
                    out.write(f"    if (kernelName == \"{kernel_name}\") {{\n")
                    
                    # Check each architecture's format
                    for arch, format_type in formats_by_arch.items():
                        compute_cap = int(arch.split('_')[1])
                        out.write(f"        if (computeCapability >= {compute_cap}) {{\n")
                        out.write(f"            return \"{format_type}\";\n")
                        out.write(f"        }}\n")
                    
                    # Default to the format of the lowest architecture
                    lowest_arch = min(formats_by_arch.keys(), key=lambda x: int(x.split('_')[1]))
                    out.write(f"        return \"{formats_by_arch[lowest_arch]}\";\n")
                    out.write(f"    }}\n")
            
            # Default fallback
            out.write("    // Default to optixir for unknown kernels\n")
            out.write("    return \"optixir\";\n")
            out.write("}\n\n")
            
            # Add function to check if a kernel is available
            out.write("// Check if a specific kernel is available\n")
            out.write("inline bool hasKernel(const std::string& kernelName) {\n")
            out.write("    static const std::vector<std::string> availableKernels = {\n")
            for kernel in sorted(all_kernels):
                out.write(f"        \"{kernel}\",\n")
            out.write("    };\n")
            out.write("    return std::find(availableKernels.begin(), availableKernels.end(), kernelName) != availableKernels.end();\n")
            out.write("}\n\n")
            
            # Add utility function for initialization
            out.write("// Static initialization function - not required but good practice\n")
            out.write("inline void initialize() {\n")
            out.write("    // This function exists just to ensure the embedded data is properly initialized\n")
            out.write("}\n\n")
            out.write("} // namespace embedded\n")
        
        # Verify file was created
        if os.path.exists(output_file):
            print(f"Successfully generated file at {output_file} (Size: {os.path.getsize(output_file)} bytes)")
        else:
            print(f"ERROR: File was not created at {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()